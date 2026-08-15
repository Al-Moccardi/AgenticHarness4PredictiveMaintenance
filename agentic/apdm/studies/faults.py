"""Fault layer for FA-PdM: phenotypes, interpretations, conditioned priors.

WHAT A "FAULT" IS HERE
----------------------
FD002 has one designed fault mode (HPC degradation, Saxena et al. 2008), but
it expresses through distinct sensor channels across units. We define fault
PHENOTYPES empirically: the unit's terminal signature = per-sensor mean
z-score (vs the healthy reference of each row's own regime) over the last
`tail` cycles before EoL. TRAIN-unit signatures are clustered (k-means,
k chosen by silhouette over {2,3,4}); each phenotype gets a deterministic,
data-grounded INTERPRETATION built only from train members: characteristic
(sensor, direction) set, physics mapping via the CMAPSS turbofan schematic,
onset statistics of the k7 degradation state, and residual-life-after-entry
priors. Interpretations are templates over measured numbers -- no free-text
generation -- so the knowledge base itself cannot hallucinate; an LLM may
later paraphrase them, but the evaluable content is fixed.

GOLD AND LEAKAGE
----------------
A test unit's gold phenotype is its own FUTURE terminal signature assigned to
the nearest train centroid. Diagnosis inputs at (unit, cycle) use only rows
<= cycle; the gold uses only rows in the terminal tail. Guarded in
smoke_test (F-series).

CIRCULARITY GUARD
-----------------
Random-forest importances from the EDA notebook are NEVER targets; they are
not used at all here. Twin baselines for diagnosis are (a) a no-learning
rule: current-window mean-z -> nearest phenotype centroid, and (b) logistic
regression on the shared 75 features. Any agent must beat (a) and (b) to
claim diagnostic value -- the kNN-twin lesson applied to RCA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data import FD002, SENSORS, W

# CMAPSS turbofan schematic mapping (Saxena et al. 2008, Fig. 1 lineage).
SENSOR_PHYSICS: Dict[str, Tuple[str, str]] = {
    "T24": ("LPC", "LPC outlet temperature"),
    "T30": ("HPC", "HPC outlet temperature"),
    "T50": ("LPT", "LPT outlet temperature"),
    "P30": ("HPC", "HPC outlet pressure"),
    "Ps30": ("HPC", "HPC outlet static pressure"),
    "phi": ("core", "fuel flow / Ps30 ratio"),
    "Nf": ("fan", "physical fan speed"),
    "NRf": ("fan", "corrected fan speed"),
    "Nc": ("core", "physical core speed"),
    "NRc": ("core", "corrected core speed"),
    "BPR": ("fan", "bypass ratio"),
    "htBleed": ("bleed", "bleed enthalpy"),
    "W31": ("HPT", "HPT coolant bleed"),
    "W32": ("LPT", "LPT coolant bleed"),
}


@dataclass
class Phenotype:
    pid: int
    name: str
    n_train_units: int
    centroid_z: Dict[str, float]
    signature: List[Tuple[str, str]]          # top (sensor, direction)
    components: List[str]                      # implicated subsystems, ranked
    onset_lead_median: float                   # EoL - state entry, train
    onset_lead_iqr: Tuple[float, float]
    residual_after_entry_median: float
    residual_after_entry_iqr: Tuple[float, float]
    interpretation: str = ""


@dataclass
class FaultLayer:
    tail: int
    k: int
    silhouette: float
    sensor_order: List[str]
    centroids: np.ndarray                      # (k, n_sensors), z-space
    phenotypes: List[Phenotype]
    train_unit_phenotype: Dict[int, int]
    pooled_residual_median: float
    pooled_residual_iqr: Tuple[float, float]

    # ------------------------------------------------------------- assign
    def assign_z(self, zvec: np.ndarray) -> Tuple[int, float]:
        d = np.linalg.norm(self.centroids - zvec[None, :], axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def library_json(self) -> str:
        return json.dumps({
            "n_phenotypes": self.k,
            "phenotypes": [{
                "id": p.pid, "name": p.name,
                "n_train_units": p.n_train_units,
                "signature": [f"{s} {d}" for s, d in p.signature],
                "implicated_components": p.components,
                "state_onset_lead_cycles":
                    {"median": p.onset_lead_median,
                     "iqr": list(p.onset_lead_iqr)},
                "residual_life_after_state_entry":
                    {"median": p.residual_after_entry_median,
                     "iqr": list(p.residual_after_entry_iqr)},
                "interpretation": p.interpretation,
            } for p in self.phenotypes],
            "note": "statistics from TRAINING units only; residuals in raw "
                    "cycles (uncapped)"})


# --------------------------------------------------------------------- build
def terminal_z(ds: FD002, unit: int, tail: int) -> np.ndarray:
    g = ds._by_unit[unit]
    tl = g[g["cycle"] > ds.eol[unit] - tail]
    out = []
    for s in SENSORS:
        acc = [(r[s] - ds.regime_ref[int(r["h_clust"])][s][0])
               / ds.regime_ref[int(r["h_clust"])][s][1]
               for _, r in tl.iterrows()]
        out.append(float(np.mean(acc)))
    return np.asarray(out)


def current_z(ds: FD002, unit: int, cycle: int, win: int = W) -> np.ndarray:
    """Diagnosis-time analogue of terminal_z: mean z over (cycle-win, cycle].
    Uses only past rows; the no-learning diagnosis twin is nearest-centroid
    on this vector."""
    h = ds.history(unit, cycle, n=win)
    out = []
    for s in SENSORS:
        acc = [(r[s] - ds.regime_ref[int(r["h_clust"])][s][0])
               / ds.regime_ref[int(r["h_clust"])][s][1]
               for _, r in h.iterrows()]
        out.append(float(np.mean(acc)))
    return np.asarray(out)


def _direction(v: float) -> str:
    return "high" if v > 0 else "low"


def _interp(p: Phenotype) -> str:
    sig = ", ".join(f"{s} {d} (z={p.centroid_z[s]:+.1f})"
                    for s, d in p.signature)
    comp = ", ".join(p.components)
    return (f"Phenotype {p.name}: degradation expressed primarily through "
            f"{sig}. Implicated subsystems (CMAPSS schematic): {comp}. "
            f"Across {p.n_train_units} training units, the degradation state "
            f"(k=7 cluster 6) is entered a median of "
            f"{p.onset_lead_median:.0f} cycles before end of life "
            f"(IQR {p.onset_lead_iqr[0]:.0f}-{p.onset_lead_iqr[1]:.0f}); "
            f"median residual life after entry is "
            f"{p.residual_after_entry_median:.0f} cycles "
            f"(IQR {p.residual_after_entry_iqr[0]:.0f}-"
            f"{p.residual_after_entry_iqr[1]:.0f}).")


def build_fault_layer(ds: FD002, tail: int = 10, k_range=(2, 3, 4),
                      top_k_sig: int = 3, seed: int = 42,
                      cache: Optional[Path] = None) -> FaultLayer:
    if cache and cache.exists():
        import pickle
        with open(cache, "rb") as f:
            return pickle.load(f)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    units = ds.train_units
    Z = np.vstack([terminal_z(ds, u, tail) for u in units])

    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        sil = float(silhouette_score(Z, km.labels_))
        if best is None or sil > best[0]:
            best = (sil, k, km)
    sil, k, km = best

    resid_all = []
    phen: List[Phenotype] = []
    unit_ph = {int(u): int(l) for u, l in zip(units, km.labels_)}
    for pid in range(k):
        members = [u for u in units if unit_ph[u] == pid]
        cz = {s: float(v) for s, v in zip(SENSORS, km.cluster_centers_[pid])}
        top = sorted(cz.items(), key=lambda kv: -abs(kv[1]))[:top_k_sig]
        sig = [(s, _direction(v)) for s, v in top]
        comps, seen = [], set()
        for s, _ in top:
            c = SENSOR_PHYSICS[s][0]
            if c not in seen:
                comps.append(c)
                seen.add(c)
        leads, resid = [], []
        for u in members:
            e = ds.state_entered(u, ds.eol[u])
            if e is not None:
                leads.append(ds.eol[u] - e)
                resid.append(ds.eol[u] - e)
        resid_all += resid
        name = "-".join(s for s, _ in sig[:2])
        p = Phenotype(
            pid=pid, name=name, n_train_units=len(members), centroid_z=cz,
            signature=sig, components=comps,
            onset_lead_median=float(np.median(leads)) if leads else float("nan"),
            onset_lead_iqr=(float(np.quantile(leads, .25)),
                            float(np.quantile(leads, .75))) if leads else (0, 0),
            residual_after_entry_median=float(np.median(resid)) if resid else float("nan"),
            residual_after_entry_iqr=(float(np.quantile(resid, .25)),
                                      float(np.quantile(resid, .75))) if resid else (0, 0))
        p.interpretation = _interp(p)
        phen.append(p)

    layer = FaultLayer(
        tail=tail, k=k, silhouette=sil, sensor_order=list(SENSORS),
        centroids=km.cluster_centers_.copy(), phenotypes=phen,
        train_unit_phenotype=unit_ph,
        pooled_residual_median=float(np.median(resid_all)),
        pooled_residual_iqr=(float(np.quantile(resid_all, .25)),
                             float(np.quantile(resid_all, .75))))
    if cache:
        import pickle
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            import pickle as pk
            pk.dump(layer, f)
    return layer


# ------------------------------------------------------------------- golds
def gold_phenotype(ds: FD002, layer: FaultLayer, unit: int) -> int:
    """Future-derived gold for any unit: its own terminal signature assigned
    to the nearest TRAIN centroid."""
    z = terminal_z(ds, unit, layer.tail)
    return layer.assign_z(z)[0]


def gold_signature(ds: FD002, layer: FaultLayer, unit: int,
                   top_k: int = 3) -> List[Tuple[str, str]]:
    z = terminal_z(ds, unit, layer.tail)
    idx = np.argsort(-np.abs(z))[:top_k]
    return [(SENSORS[i], _direction(z[i])) for i in idx]
