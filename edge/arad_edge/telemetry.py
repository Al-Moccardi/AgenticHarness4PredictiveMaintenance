"""
telemetry.py — hardware telemetry sampler (Jetson-aware, degrades anywhere)
===========================================================================
Samples system state at a fixed interval into telemetry.jsonl. On a Jetson
(Orin) it reads the iGPU load, thermal zones and the INA3221 power rails from
sysfs; on any Linux it still records CPU and RAM, so the pipeline and the
stats code run unchanged off-device.

Fields per sample (missing on unsupported hosts -> null):
  t, cpu_pct, mem_used_mb, mem_total_mb, gpu_pct, temp_c_max, power_w

Run alongside the experiment:
    python3 telemetry.py --out telemetry.jsonl --interval 1 &
Stop with Ctrl-C / kill; edge_stats.py integrates power over time for energy.
"""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path

def _cpu_times():
    with open("/proc/stat") as f:
        p = f.readline().split()[1:8]
    v = list(map(int, p)); return sum(v), v[3]

def _mem_mb():
    tot = avail = None
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"): tot = int(line.split()[1]) // 1024
        if line.startswith("MemAvailable"): avail = int(line.split()[1]) // 1024
        if tot and avail: break
    return (tot - avail) if (tot and avail) else None, tot

def _gpu_pct():
    for p in ("/sys/devices/gpu.0/load",
              "/sys/devices/platform/gpu.0/load",
              "/sys/kernel/debug/gpu.0/load"):
        try: return int(open(p).read().strip()) / 10.0
        except Exception: pass
    return None

def _temp_max():
    ts = []
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try: ts.append(int(open(p).read().strip()) / 1000.0)
        except Exception: pass
    return max(ts) if ts else None

def _power_w():
    """Orin INA3221: hwmon curr*_input (mA) + in*_input (mV) per rail."""
    tot_mw = 0.0; found = False
    for hw in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*") + \
              glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = open(Path(hw) / "name").read().strip().lower()
        except Exception:
            continue
        if "ina3221" not in name and "ina" not in name:
            continue
        for ci in glob.glob(str(Path(hw) / "curr*_input")):
            ch = Path(ci).name.replace("curr", "").replace("_input", "")
            vi = Path(hw) / f"in{ch}_input"
            try:
                ma = int(open(ci).read().strip())
                mv = int(open(vi).read().strip())
                tot_mw += ma * mv / 1000.0; found = True
            except Exception:
                pass
    return round(tot_mw / 1000.0, 2) if found else None

def main() -> None:
    ap = argparse.ArgumentParser()
    from . import paths
    paths.ensure_results()
    ap.add_argument("--out", default=str(paths.RESULTS / "telemetry.jsonl"))
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after N seconds (0 = until killed)")
    a = ap.parse_args()
    out = Path(a.out); t_end = time.time() + a.duration if a.duration else None
    pt, pi = _cpu_times()
    with out.open("a") as f:
        try:
            while True:
                time.sleep(a.interval)
                ct, ci = _cpu_times()
                cpu = 100.0 * (1 - (ci - pi) / max(ct - pt, 1)); pt, pi = ct, ci
                used, tot = _mem_mb()
                rec = {"t": time.time(), "cpu_pct": round(cpu, 1),
                       "mem_used_mb": used, "mem_total_mb": tot,
                       "gpu_pct": _gpu_pct(), "temp_c_max": _temp_max(),
                       "power_w": _power_w()}
                f.write(json.dumps(rec) + "\n"); f.flush()
                if t_end and time.time() > t_end: break
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
