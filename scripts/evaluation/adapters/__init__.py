"""Method adapters: load per-method predictions into a canonical DataFrame."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from .fice import FiceAdapter
from .gcare import GcareAdapter
from .gnce import GnceAdapter
from .lmkg import LmkgAdapter
from .lss import LssAdapter


class MethodAdapter(Protocol):
    name: str

    def load(self, root: Path) -> pd.DataFrame: ...


ADAPTER_REGISTRY: dict[str, MethodAdapter] = {
    "fice": FiceAdapter(),
    "gnce": GnceAdapter(),
    "wj": GcareAdapter("wj"),
    "cset": GcareAdapter("cset"),
    "lmkg": LmkgAdapter(),
    "lss": LssAdapter(),
    "sumrdf": GcareAdapter("sumrdf"),
}

__all__ = [
    "MethodAdapter",
    "ADAPTER_REGISTRY",
    "FiceAdapter",
    "GnceAdapter",
    "GcareAdapter",
    "LmkgAdapter",
    "LssAdapter",
]
