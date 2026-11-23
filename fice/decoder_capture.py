"""Capture of the inputs and outputs of the FICE decoder (``model.query_gnn``).

Kickoff WS 26/27, Sec. 5.2: "All analyses consume records captured at the
decoder's input." This module provides that single interface:

* a forward **pre-hook** on ``model.query_gnn`` that stores copies of
  ``batch.x``, ``batch.edge_attr``, ``batch.edge_index``, ``batch.batch`` and
  ``graph_size``. ``FICE.forward`` overwrites ``batch.x`` / ``batch.edge_attr``
  in place before it calls the decoder, so the copies are taken *inside* the
  hook (exactly what the decoder sees)
* a forward **hook** on ``model.query_gnn`` that stores the raw decoder output;
* a forward hook on the top-level model that stores the value ``FICE.forward``
  returns (the predicted log-cardinality). With ``prediction_mode: log-selectivity``
  this differs from the raw decoder
  output (log upper bound + logsigmoid(output)); Sec. 5.1 requires y_hat_FICE
  as the regression target, but upper bound isn't trained;
* :func:`verify_reproduction`, the one-off check: re-run``query_gnn`` in eval
  mode on the captured inputs and compare with the stored outputs.

Note: ``query_gnn`` is called with keyword arguments only
(``self.query_gnn(graph_size=..., batch=...)``). A plain pre-hook only sees
positional args, so the hooks are registered with ``with_kwargs=True``
(torch >= 2.0).

Usage (e.g. inside an evaluation loop as in ``scripts/train.py``)::

    from fice.decoder_capture import DecoderCapture, verify_reproduction, save_records

    model.eval()
    with torch.no_grad(), DecoderCapture(model) as cap:
        for ... in loader:
            model(data=graph, relation_graph=graph, batch=query_batch, graph_size=graph_size)

    report = verify_reproduction(model.query_gnn, cap.records)   # raises on mismatch
    save_records(cap.records, "records/fano_seed0.pt", meta={"checkpoint": str(ckpt_path), "seed": 0})

Relation embeddings for geometric analysis: ``record.relation_embeddings()``
(the columns after the 3 count features of ``edge_attr``). Parameters from
``state_dict`` are never embeddings (Sec. 4.2 / 5.2).

Capture in eval mode and without autocast: dropout or half precision make the
reproduction check fail by construction.
"""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch import Tensor, nn

__all__ = [
    "DecoderRecord",
    "DecoderCapture",
    "replay",
    "verify_reproduction",
    "save_records",
    "load_records",
]

NUM_COUNT_FEATURES = 3
_FORMAT_VERSION = 1


def _snapshot(value: Any) -> Any:
    """Detached copy on CPU, independent of later in-place changes."""
    if value is None:
        return None
    if isinstance(value, Tensor):
        return value.detach().clone().cpu()
    if isinstance(value, (int, float)):
        return torch.tensor([value])
    return value


def _per_query(value: Optional[Tensor], g: int, n: int) -> Optional[Tensor]:
    if value is None or not isinstance(value, Tensor):
        return None
    if value.dim() > 0 and value.size(0) == n:
        return value[g]
    return None


@dataclass
class DecoderRecord:
    """One call of ``query_gnn`` (= one mini-batch of query graphs)."""

    x: Tensor                      # [num_nodes, 3 + d]  node features [c1,c2,c3 | e_v]
    edge_attr: Tensor              # [num_edges, 3 + d]  edge features [c1,c2,c3 | h_r]
    edge_index: Tensor             # [2, num_edges]
    batch: Tensor                  # [num_nodes]         query index per node
    graph_size: Optional[Tensor]   # size of the data graph (as passed to query_gnn)
    output: Optional[Tensor] = None      # raw query_gnn output [num_queries]
    prediction: Optional[Tensor] = None  # FICE.forward return value = y_hat_FICE (log-cardinality)
    extras: Dict[str, Tensor] = field(default_factory=dict)  # e.g. y (true cardinality), edge_type
    call_index: int = 0
    training: bool = False         # module was in training mode during capture (replay unreliable)

    # ------------------------------------------------------------- accessors
    @property
    def num_queries(self) -> int:
        if self.output is not None and self.output.dim() > 0:
            return int(self.output.size(0))
        return int(self.batch.max().item()) + 1 if self.batch.numel() else 0

    @property
    def y(self) -> Optional[Tensor]:
        return self.extras.get("y")

    def relation_embeddings(self, d: Optional[int] = None) -> Tensor:
        """Per-edge relation embeddings h_r (last ``d`` columns of ``edge_attr``)."""
        if d is None:
            return self.edge_attr[:, NUM_COUNT_FEATURES:]
        return self.edge_attr[:, -d:]

    def node_embeddings(self, d: Optional[int] = None) -> Tensor:
        if d is None:
            return self.x[:, NUM_COUNT_FEATURES:]
        return self.x[:, -d:]

    def edge_batch(self) -> Tensor:
        """Query index per edge."""
        return self.batch[self.edge_index[0]]

    def to_data(self, device: Optional[Union[str, torch.device]] = None):
        """Rebuild a PyG object with the attributes ``query_gnn`` reads."""
        from torch_geometric.data import Data

        data = Data(x=self.x.clone(), edge_index=self.edge_index.clone(),
                    edge_attr=self.edge_attr.clone(), batch=self.batch.clone())
        return data.to(device) if device is not None else data

    def split_queries(self) -> List[Dict[str, Any]]:
        """Split the batch into one dict per query graph (local node indices)."""
        n = self.num_queries
        eb = self.edge_batch()
        result = []
        for g in range(n):
            node_idx = (self.batch == g).nonzero(as_tuple=False).view(-1)
            remap = torch.full((self.x.size(0),), -1, dtype=torch.long)
            remap[node_idx] = torch.arange(node_idx.numel())
            emask = eb == g
            item = {
                "x": self.x[node_idx],
                "edge_attr": self.edge_attr[emask],
                "edge_index": remap[self.edge_index[:, emask]],
                "output": _per_query(self.output, g, n),
                "prediction": _per_query(self.prediction, g, n),
            }
            for k, v in self.extras.items():
                if v.dim() > 0 and v.size(0) == self.edge_index.size(1):
                    item[k] = v[emask]
                elif v.dim() > 0 and v.size(0) == self.x.size(0):
                    item[k] = v[node_idx]
                else:
                    item[k] = _per_query(v, g, n)
            result.append(item)
        return result

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecoderRecord":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _bind(module: nn.Module, args: tuple, kwargs: dict) -> Dict[str, Any]:
    try:
        return dict(inspect.signature(module.forward).bind(*args, **kwargs).arguments)
    except TypeError:
        bound = dict(kwargs)
        if args:
            bound.setdefault("batch", args[0])
        return bound


class DecoderCapture:
    """Context manager that records every call of ``model.query_gnn``.

    Args:
        model: a FICE model (anything with a ``query_gnn`` submodule), or the
            decoder itself if ``decoder_only=True``.
        extra_attrs: further attributes of the query batch to store if present
            (default: ``y`` = true cardinality, ``edge_type`` = predicate ids).
        capture_prediction: also hook the top-level model to store y_hat_FICE.
        dedupe_repeat: with ``FICE.forward(..., repeat>1)`` the decoder runs
            several times on the same input; keep only the first call.
    """

    def __init__(self, model: nn.Module, extra_attrs: Sequence[str] = ("y", "edge_type"),
                 capture_prediction: bool = True, dedupe_repeat: bool = True,
                 decoder_only: bool = False):
        if decoder_only:
            self.model, self.decoder = None, model
        else:
            if not hasattr(model, "query_gnn"):
                raise AttributeError("model has no attribute 'query_gnn'")
            self.model, self.decoder = model, model.query_gnn
        self.extra_attrs = tuple(extra_attrs)
        self.capture_prediction = capture_prediction and self.model is not None
        self.dedupe_repeat = dedupe_repeat
        self.records: List[DecoderRecord] = []
        self._handles: list = []
        self._open: List[Optional[DecoderRecord]] = []   # pre-hook -> forward hook pairing
        self._outer_depth = 0
        self._outer_records: List[DecoderRecord] = []
        self._warned_training = False

    # ---------------------------------------------------------------- hooks
    def _decoder_pre_hook(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        if self.dedupe_repeat and self._outer_depth > 0 and self._outer_records:
            self._open.append(None)          # repeated call within one FICE.forward
            return
        bound = _bind(module, args, kwargs)
        batch = bound.get("batch")
        if batch is None:
            raise RuntimeError("DecoderCapture: query_gnn was called without a 'batch' argument")
        graph_size = bound.get("graph_size", bound.get("N"))

        if module.training and not self._warned_training:
            warnings.warn("DecoderCapture: query_gnn is in training mode; dropout makes "
                          "the reproduction check unreliable. Call model.eval() first.")
            self._warned_training = True

        extras = {}
        for name in self.extra_attrs:
            value = getattr(batch, name, None)
            if isinstance(value, Tensor):
                extras[name] = _snapshot(value)

        rec = DecoderRecord(
            x=_snapshot(batch.x),
            edge_attr=_snapshot(batch.edge_attr),
            edge_index=_snapshot(batch.edge_index),
            batch=_snapshot(batch.batch),
            graph_size=_snapshot(graph_size),
            extras=extras,
            call_index=len(self.records),
            training=module.training,
        )
        self.records.append(rec)
        self._open.append(rec)
        if self._outer_depth > 0:
            self._outer_records.append(rec)

    def _decoder_hook(self, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        rec = self._open.pop() if self._open else None
        if rec is not None:
            rec.output = _snapshot(output[0] if isinstance(output, tuple) else output)

    def _model_pre_hook(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        if self._outer_depth == 0:
            self._outer_records = []
        self._outer_depth += 1

    def _model_hook(self, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        self._outer_depth -= 1
        if self._outer_depth == 0:
            pred = output[0] if isinstance(output, tuple) else output   # repeat>1 returns (pred, time)
            for rec in self._outer_records:
                rec.prediction = _snapshot(pred)
            self._outer_records = []

    # ------------------------------------------------------- context manager
    def attach(self) -> "DecoderCapture":
        if self._handles:
            return self
        self._handles.append(self.decoder.register_forward_pre_hook(self._decoder_pre_hook, with_kwargs=True))
        self._handles.append(self.decoder.register_forward_hook(self._decoder_hook, with_kwargs=True))
        if self.capture_prediction:
            self._handles.append(self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
            self._handles.append(self.model.register_forward_hook(self._model_hook, with_kwargs=True))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._open = []
        self._outer_depth = 0
        self._outer_records = []

    def __enter__(self) -> "DecoderCapture":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    def clear(self) -> None:
        self.records = []


# ------------------------------------------------------------------ replay
@torch.no_grad()
def replay(query_gnn: nn.Module, record: DecoderRecord,
           device: Optional[Union[str, torch.device]] = None) -> Tensor:
    """Run ``query_gnn`` in eval mode on a captured input; returns the output on CPU."""
    if device is None:
        try:
            device = next(query_gnn.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    data = record.to_data(device)
    gs = record.graph_size.to(device) if isinstance(record.graph_size, Tensor) else record.graph_size
    params = inspect.signature(query_gnn.forward).parameters
    kw = {"batch": data}
    if "graph_size" in params:
        kw["graph_size"] = gs
    elif "N" in params:
        kw["N"] = gs
    was_training = query_gnn.training
    query_gnn.eval()
    try:
        out = query_gnn(**kw)
    finally:
        query_gnn.train(was_training)
    if isinstance(out, tuple):
        out = out[0]
    return out.detach().cpu()


def verify_reproduction(query_gnn: nn.Module, records: Iterable[DecoderRecord],
                        atol: float = 1e-5, rtol: float = 1e-5,
                        raise_on_fail: bool = True) -> Dict[str, Any]:
    """Sec. 5.2 check: replayed outputs must match the stored outputs.

    Returns a report with ``n_records``, ``n_values``, ``max_abs_diff``,
    ``max_rel_diff`` and the indices of failing records.
    """
    n_records = n_values = 0
    max_abs = max_rel = 0.0
    failed: List[int] = []
    for i, rec in enumerate(records):
        if rec.output is None:
            continue
        if rec.training:
            warnings.warn(f"record {i} was captured in training mode")
        if rec.output.dtype in (torch.float16, torch.bfloat16):
            warnings.warn(f"record {i} has dtype {rec.output.dtype} (autocast?); expect larger deviations")
        out = replay(query_gnn, rec).to(torch.float64)
        ref = rec.output.to(torch.float64)
        if out.shape != ref.shape:
            failed.append(i)
            continue
        diff = (out - ref).abs()
        n_records += 1
        n_values += ref.numel()
        if diff.numel():
            max_abs = max(max_abs, diff.max().item())
            max_rel = max(max_rel, (diff / ref.abs().clamp(min=1e-12)).max().item())
        if not torch.allclose(out, ref, atol=atol, rtol=rtol):
            failed.append(i)
    report = {"n_records": n_records, "n_values": n_values, "max_abs_diff": max_abs,
              "max_rel_diff": max_rel, "failed": failed, "atol": atol, "rtol": rtol,
              "ok": not failed and n_records > 0}
    if raise_on_fail and not report["ok"]:
        raise AssertionError(f"decoder reproduction check failed: {report}")
    return report


# ------------------------------------------------------------- persistence
def save_records(records: Sequence[DecoderRecord], path: Union[str, Path],
                 meta: Optional[Dict[str, Any]] = None) -> Path:
    """Save records (checkpoints/records belong on shared storage, not in git)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": _FORMAT_VERSION, "meta": dict(meta or {}),
                "records": [r.to_dict() for r in records]}, path)
    return path


def load_records(path: Union[str, Path]):
    """Returns ``(records, meta)``."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    return [DecoderRecord.from_dict(d) for d in blob["records"]], blob.get("meta", {})
