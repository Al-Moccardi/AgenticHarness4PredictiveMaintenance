"""Edge-hardware cost model (NVIDIA Jetson Orin Nano class) for apdm.

WHAT THIS IS: a first-order ANALYTICAL model of quantized-SLM inference cost
on Jetson-class unified-memory devices. Decode is memory-bandwidth bound
(every token streams the weights + KV cache); prefill is compute bound; the
KV cache competes with the weights for the same unified memory, which is why
context length is a HARDWARE decision on an 8 GB module.

    t_prefill = 2 * P * (n_in - n_cached) / (F_peak * eta_c)
    t_decode  = n_out * (B_w + B_kv(ctx)) / (BW_peak * eta_b)
    E         = (t_prefill + t_decode) * P_active

WHAT THIS IS NOT: a measurement. Every derived quantity carries the `sim_`
prefix; `provenance()` prints every constant including the derivation
TFLOPS_fp16 = TOPS_int8_sparse / 4 and `efficiency_calibrated: false` until
`calibrate()` is fed measured (n_in, n_out, seconds) triples FROM THE REAL
DEVICE. Wall-clock on the experiment host is recorded separately and never
mixed with modelled edge time. Token counts, however, are REAL: the Ollama
backend logs prompt_eval_count / eval_count per call, and the model consumes
those; the chars/3.6 heuristic is only the fallback.

Verify the DEVICES constants against the datasheet revision you cite; NVIDIA
revised Orin Nano figures across JetPack releases (the 'Super' mode).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

QUANT_BPW: Dict[str, float] = {"Q4_0": 4.55, "Q4_K_M": 4.85, "Q5_K_M": 5.70,
                               "Q8_0": 8.50, "FP16": 16.0}


@dataclass
class DeviceProfile:
    name: str
    mem_total_gb: float
    os_reserved_gb: float
    mem_bandwidth_gbs: float        # theoretical peak
    int8_tops_sparse: float         # published; fp16 dense derived as /4
    active_power_w: float
    idle_power_w: float
    bw_efficiency: float = 0.60     # CALIBRATE on the device
    compute_efficiency: float = 0.20
    calibrated: bool = False

    @property
    def fp16_tflops_dense(self) -> float:
        return self.int8_tops_sparse / 4.0

    @property
    def usable_mem_gb(self) -> float:
        return self.mem_total_gb - self.os_reserved_gb


DEVICES: Dict[str, DeviceProfile] = {
    "orin_nano_8gb": DeviceProfile(
        "Jetson Orin Nano 8GB (15W)", 8.0, 1.6, 68.0, 40.0, 15.0, 2.8),
    "orin_nano_8gb_super": DeviceProfile(
        "Jetson Orin Nano Super 8GB (25W, JetPack 6.2)", 8.0, 1.6, 102.0,
        67.0, 25.0, 3.4),
    "orin_nx_16gb": DeviceProfile(
        "Jetson Orin NX 16GB (25W)", 16.0, 1.8, 102.4, 100.0, 25.0, 3.6),
}


@dataclass
class ModelProfile:
    name: str
    params: float
    n_layers: int
    n_kv_heads: int
    head_dim: int


MODELS: Dict[str, ModelProfile] = {
    "llama3.2:1b": ModelProfile("llama3.2:1b", 1.24e9, 16, 8, 64),
    "gemma2:2b":   ModelProfile("gemma2:2b",   2.61e9, 26, 4, 256),
    "qwen2.5:3b":  ModelProfile("qwen2.5:3b",  3.09e9, 36, 2, 128),
    "llama3.2:3b": ModelProfile("llama3.2:3b", 3.21e9, 28, 8, 128),
    "phi3":        ModelProfile("phi3:3.8b",   3.82e9, 32, 32, 96),
    "mistral":     ModelProfile("mistral:7b",  7.24e9, 32, 8, 128),
    "qwen2.5:7b":  ModelProfile("qwen2.5:7b",  7.62e9, 28, 4, 128),
    "llama3":      ModelProfile("llama3:8b",   8.03e9, 32, 8, 128),
    "llama3.1":    ModelProfile("llama3.1:8b", 8.03e9, 32, 8, 128),
}


def resolve_model(name: str) -> ModelProfile:
    key = (name or "").lower()
    if key in MODELS:
        return MODELS[key]
    base = key.split(":")[0]
    for k in MODELS:
        if k.split(":")[0] == base:
            return MODELS[k]
    print(f"[hardware] WARNING: no profile for '{name}', using llama3:8b "
          f"stand-in; add it to MODELS before reporting.")
    return MODELS["llama3"]


CHARS_PER_TOKEN = 3.6


def count_tokens(text: str) -> int:
    return max(1, round(len(text or "") / CHARS_PER_TOKEN))


class CostModel:
    def __init__(self, device: str = "orin_nano_8gb", model: str = "llama3",
                 quant: str = "Q4_K_M", kv_bits: int = 16,
                 num_ctx: int = 8192):
        if device not in DEVICES:
            raise ValueError(f"unknown device '{device}'; have "
                             f"{sorted(DEVICES)}")
        self.device_key = device
        self.d = DEVICES[device]
        self.m = resolve_model(model)
        self.quant = quant
        self.kv_bits = kv_bits
        self.num_ctx = num_ctx

    # -------------------------------------------------------------- memory
    def weight_bytes(self) -> float:
        return self.m.params * QUANT_BPW[self.quant] / 8.0

    def kv_bytes_per_token(self) -> float:
        return 2 * self.m.n_layers * self.m.n_kv_heads * self.m.head_dim \
            * self.kv_bits / 8.0

    def memory_footprint(self, ctx: Optional[int] = None) -> Dict:
        ctx = ctx or self.num_ctx
        w = self.weight_bytes() / 1e9
        kv = self.kv_bytes_per_token() * ctx / 1e9
        return {"weights_gb": round(w, 3), "kv_gb": round(kv, 3),
                "total_gb": round(w + kv, 3),
                "usable_gb": round(self.d.usable_mem_gb, 3),
                "fits": (w + kv) <= self.d.usable_mem_gb}

    def max_context(self) -> int:
        free = self.d.usable_mem_gb * 1e9 - self.weight_bytes()
        return max(0, int(free / self.kv_bytes_per_token()))

    # ------------------------------------------------------------- latency
    def estimate(self, n_in: int, n_out: int, n_cached: int = 0) -> Dict:
        eff_bw = self.d.mem_bandwidth_gbs * 1e9 * self.d.bw_efficiency
        eff_fl = self.d.fp16_tflops_dense * 1e12 * self.d.compute_efficiency
        n_pre = max(int(n_in) - max(int(n_cached), 0), 0)
        prefill = 2.0 * self.m.params * max(n_pre, 1) / eff_fl
        kv_mid = self.kv_bytes_per_token() * (n_in + max(n_out, 1) / 2.0)
        decode = max(n_out, 0) * (self.weight_bytes() + kv_mid) / eff_bw
        total = prefill + decode
        return {"sim_edge_s": total, "sim_prefill_s": prefill,
                "sim_decode_s": decode,
                "sim_energy_j": total * self.d.active_power_w,
                "sim_tok_per_s": (n_out / decode) if decode > 0 else 0.0,
                "sim_fits": self.memory_footprint()["fits"]}

    # --------------------------------------------------------- calibration
    def calibrate(self, samples) -> Dict[str, float]:
        """samples: measured (n_in, n_out, seconds) triples FROM THE DEVICE."""
        import numpy as np
        A, y = [], []
        for n_in, n_out, secs in samples:
            kv_mid = self.kv_bytes_per_token() * (n_in + n_out / 2.0)
            A.append([2.0 * self.m.params * n_in,
                      n_out * (self.weight_bytes() + kv_mid)])
            y.append(secs)
        coef, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(y), rcond=None)
        if coef[0] <= 0 or coef[1] <= 0:
            raise RuntimeError("non-physical calibration fit")
        self.d.compute_efficiency = (1.0 / coef[0]) / (
            self.d.fp16_tflops_dense * 1e12)
        self.d.bw_efficiency = (1.0 / coef[1]) / (
            self.d.mem_bandwidth_gbs * 1e9)
        self.d.calibrated = True
        return {"compute_efficiency": self.d.compute_efficiency,
                "bw_efficiency": self.d.bw_efficiency, "n": len(samples)}

    # ---------------------------------------------------------- provenance
    def provenance(self) -> Dict:
        mem = self.memory_footprint()
        return {"device": self.d.name, "model": self.m.name,
                "quant": self.quant, "num_ctx": self.num_ctx,
                "mem_bandwidth_gbs_peak": self.d.mem_bandwidth_gbs,
                "int8_tops_sparse_published": self.d.int8_tops_sparse,
                "fp16_tflops_dense_derived": round(self.d.fp16_tflops_dense, 2),
                "fp16_derivation": "TOPS_int8_sparse/2(sparse->dense)/2(int8->fp16)",
                "bw_efficiency": self.d.bw_efficiency,
                "compute_efficiency": self.d.compute_efficiency,
                "efficiency_calibrated": self.d.calibrated,
                "active_power_w": self.d.active_power_w,
                **mem, "max_context_that_fits": self.max_context(),
                "token_source": "ollama prompt_eval_count/eval_count when "
                                "available, else chars/3.6",
                "DISCLAIMER": "ANALYTICAL MODEL, NOT MEASUREMENT. Report as "
                              "estimated; calibrate on the target device "
                              "before publication."}


def feasibility_table(models, device: str = "orin_nano_8gb",
                      num_ctx: int = 8192):
    rows = []
    for m in models:
        cm = CostModel(device, m, num_ctx=num_ctx)
        f = cm.memory_footprint()
        rows.append({"model": m, **f, "max_ctx": cm.max_context()})
    return rows
