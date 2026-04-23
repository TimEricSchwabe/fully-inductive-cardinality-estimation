"""Canonical schema shared across method adapters."""
from __future__ import annotations

CANONICAL_COLUMNS = [
    "method",
    "graph",
    "query_dataset",
    "query_type",
    "prediction",
    "true_value",
]

EPSILON = 1.0

CARDINALITY_BIN_EDGES = [0, 10, 100, 1000, 100_000, 1_000_000, float("inf")]
CARDINALITY_BIN_LABELS = ["<10", "10-100", "100-1k", "1k-100k", "100k-1M", "≥1M"]
