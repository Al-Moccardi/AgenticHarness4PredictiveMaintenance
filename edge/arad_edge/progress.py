"""One progress abstraction used everywhere. Prefers tqdm; if it is not
installed, falls back to a lightweight carriage-return bar so the Jetson run
still shows live progress with zero hard dependencies."""
from __future__ import annotations

import sys
import time
from typing import Iterable, Optional

try:
    from tqdm import tqdm as _tqdm
    _HAVE_TQDM = True
except Exception:  # noqa: BLE001
    _HAVE_TQDM = False


class _Fallback:
    def __init__(self, total: Optional[int], desc: str, unit: str):
        self.total = total or 0
        self.desc = desc
        self.unit = unit
        self.n = 0
        self.t0 = time.time()
        self._last = 0.0
        self._render()

    def _render(self, final: bool = False) -> None:
        el = time.time() - self.t0
        rate = self.n / el if el > 0 else 0.0
        if self.total:
            frac = self.n / self.total
            bar = ("#" * int(30 * frac)).ljust(30)
            eta = (self.total - self.n) / rate if rate > 0 else 0
            msg = (f"\r{self.desc} |{bar}| {self.n}/{self.total} "
                   f"({frac:5.1%}) {rate:5.2f}{self.unit}/s "
                   f"ETA {eta/60:4.1f}m")
        else:
            msg = (f"\r{self.desc} {self.n} {self.unit} "
                   f"{rate:5.2f}{self.unit}/s {el:5.0f}s")
        sys.stderr.write(msg)
        sys.stderr.flush()
        if final:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def update(self, k: int = 1) -> None:
        self.n += k
        now = time.time()
        if now - self._last > 0.2 or (self.total and self.n >= self.total):
            self._render()
            self._last = now

    def set_postfix_str(self, s: str) -> None:
        self.desc_suffix = s

    def close(self) -> None:
        self._render(final=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def bar(total: Optional[int] = None, desc: str = "", unit: str = "it"):
    """Return a progress bar object exposing .update()/.close()/.set_postfix_str()."""
    if _HAVE_TQDM:
        return _tqdm(total=total, desc=desc, unit=unit,
                     dynamic_ncols=True, file=sys.stderr)
    return _Fallback(total, desc, unit)


def track(it: Iterable, total: Optional[int] = None, desc: str = "",
          unit: str = "it"):
    b = bar(total=total, desc=desc, unit=unit)
    try:
        for x in it:
            yield x
            b.update(1)
    finally:
        b.close()
