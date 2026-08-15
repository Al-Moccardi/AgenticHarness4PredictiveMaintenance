"""Step 2: the vector store -- fleet units as a semantic knowledge base.

TIOT's Fig. 9 RAG (embedding model + vector DB over interpretations), made a
first-class, testable component. Records are interpretation texts -- the 65
real train ones plus everything gen_interpretations produces -- embedded and
searched by cosine similarity. The query at inference time is the current
sensor summary (grounded text from the shared features), so retrieval asks:
"which past interpreted anomalies read like NOW?"

Two embedders:
  ollama  nomic-embed-text (274 MB, Jetson-deployable; TIOT-consistent)
  hash    deterministic offline fallback: L2-normalised hashed character
          n-grams. NOT semantic -- exists so the pipeline is testable
          without models and so smoke tests are reproducible. Never report
          results from it.

Leakage: records are TRAIN units only (constructor filter, guard V1);
`exclude_unit` removes the query unit when it happens to be a train unit.

This gives the pattern grid its retrieval-representation axis: z-space kNN
(interpret_kb) vs semantic retrieval over generated text (here) -- with
z-kNN as the non-language twin the semantic store must beat to justify the
generation cost.

  python -m apdm.vector_store --build --embedder ollama     # on your box
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .data import FD002

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = ROOT / "data" / "vector_store"
SOURCES = [ROOT / "data" / "interpretations",
           ROOT / "data" / "interpretations_generated"]


# ---------------------------------------------------------------- embedders
class HashEmbedder:
    name = "hash-3gram-256d"
    dim = 256

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = (t or "").lower()
            for j in range(len(t) - 2):
                h = int(hashlib.md5(t[j:j + 3].encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i]) or 1.0
            out[i] /= n
        return out


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text"):
        import ollama
        self._c = ollama
        self.name = f"ollama:{model}"
        self.model = model
        self.dim = None

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            r = self._c.embeddings(model=self.model, prompt=t[:2000])
            v = np.asarray(r["embedding"], dtype=np.float32)
            v /= (np.linalg.norm(v) or 1.0)
            vecs.append(v)
        self.dim = len(vecs[0])
        return np.vstack(vecs)


def get_embedder(name: str):
    return OllamaEmbedder() if name == "ollama" else HashEmbedder()


# -------------------------------------------------------------------- store
class VectorStore:
    def __init__(self, embeddings: np.ndarray, meta: List[Dict],
                 embedder_name: str):
        self.E = embeddings
        self.meta = meta
        self.embedder_name = embedder_name

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, ds: FD002, embedder, sources: Optional[List[Path]] = None,
              train_only: bool = True) -> "VectorStore":
        texts, meta = [], []
        train = set(ds.train_units)
        for src in (sources or SOURCES):
            if not src.exists():
                continue
            for f in sorted(list(src.glob("unit_*.json"))
                            + list(src.glob("unit_*.jsonl"))):
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    u = int(r["unit_ID"])
                    if train_only and u not in train:
                        continue
                    txt = str(r.get("interpretation", "")).strip()
                    if len(txt) < 40:
                        continue
                    cyc = int(r.get("cycle", r.get("cycles", 0)))
                    meta.append({"unit": u, "cycle": cyc,
                                 "rul_then": ds.rul(u, cyc),
                                 "source": r.get("source", "real"),
                                 "gravity": r.get("gravity"),
                                 "text": txt[:900]})
                    texts.append(txt)
        if not texts:
            raise RuntimeError("no interpretation records found to index")
        E = embedder.embed(texts)
        return cls(E, meta, embedder.name)

    # ------------------------------------------------------------- persist
    def save(self, path: Path = STORE_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "embeddings.npz", E=self.E)
        (path / "meta.jsonl").write_text(
            "\n".join(json.dumps(m) for m in self.meta))
        (path / "info.json").write_text(json.dumps(
            {"embedder": self.embedder_name, "n": len(self.meta)}))

    @classmethod
    def load(cls, path: Path = STORE_DIR) -> "VectorStore":
        E = np.load(path / "embeddings.npz")["E"]
        meta = [json.loads(l) for l in
                (path / "meta.jsonl").read_text().splitlines() if l.strip()]
        info = json.loads((path / "info.json").read_text())
        return cls(E, meta, info["embedder"])

    # -------------------------------------------------------------- search
    def search(self, query_vec: np.ndarray, k: int = 4,
               exclude_unit: Optional[int] = None) -> List[Dict]:
        q = query_vec / (np.linalg.norm(query_vec) or 1.0)
        sims = self.E @ q
        order = np.argsort(-sims)
        out = []
        for i in order:
            m = self.meta[int(i)]
            if exclude_unit is not None and m["unit"] == exclude_unit:
                continue
            out.append({**m, "similarity": round(float(sims[int(i)]), 3)})
            if len(out) >= k:
                break
        return out

    def observation(self, embedder, query_text: str, k: int = 4,
                    exclude_unit: Optional[int] = None) -> str:
        qv = embedder.embed([query_text])[0]
        recs = self.search(qv, k=k, exclude_unit=exclude_unit)
        return json.dumps({
            "k": len(recs), "embedder": self.embedder_name,
            "note": "semantic retrieval over TRAIN-fleet interpretation "
                    "texts; rul_then is what actually remained",
            "precedents": [{k_: v for k_, v in r.items()} for r in recs]})


# -------------------------------------------------------------------- CLI
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--embedder", default="hash", choices=["hash", "ollama"])
    a = ap.parse_args()
    ds = FD002(seed=42)
    if a.build:
        emb = get_embedder(a.embedder)
        vs = VectorStore.build(ds, emb)
        vs.save()
        src = {}
        for m in vs.meta:
            src[m["source"]] = src.get(m["source"], 0) + 1
        print(f"[store] built: {len(vs.meta)} records ({src}), "
              f"embedder={vs.embedder_name}, dim={vs.E.shape[1]} -> "
              f"{STORE_DIR}")
        if a.embedder == "hash":
            print("[store] WARNING: hash embedder is a pipeline-test "
                  "fallback, not semantic; rebuild with --embedder ollama "
                  "before any reported run.")


if __name__ == "__main__":
    main()
