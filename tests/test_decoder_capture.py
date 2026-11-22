"""Tests for fice/decoder_capture.py (kickoff Sec. 5.2).

Run:  python -m pytest tests/test_decoder_capture.py -q
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch import nn
from torch_geometric.data import Batch, Data

from fice.decoder_capture import DecoderCapture, load_records, replay, save_records, verify_reproduction
from fice.gnce.models import TripleModel

D = 8
FEAT = D + 3


def make_star(k: int, seed: int) -> Data:
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.stack([torch.zeros(k, dtype=torch.long), torch.arange(1, k + 1)])
    return Data(x=torch.rand(k + 1, 3, generator=g) * 100,
                edge_index=edge_index,
                edge_attr=torch.rand(k, 3, generator=g) * 100,
                edge_type=torch.randint(0, 7, (k,), generator=g),
                y=torch.tensor([float(seed + 1)]))


def make_batch():
    return Batch.from_data_list([make_star(k, s) for s, k in enumerate([2, 3, 3, 2, 4])])


class FakeFICE(nn.Module):
    """Mimics FICE.forward: replaces batch.x / batch.edge_attr in place,
    calls query_gnn with keyword args only and post-processes the output."""

    def __init__(self):
        super().__init__()
        self.query_gnn = TripleModel(feat_dim=FEAT, num_layers=2)
        self.rel = nn.Parameter(torch.randn(7, D))

    def forward(self, batch, graph_size, repeat: int = 1):
        n = batch.x.size(0)
        batch.x = torch.cat([torch.log1p(batch.x[:, :3]), torch.zeros(n, D)], dim=1)
        batch.edge_attr = torch.cat([torch.log1p(batch.edge_attr[:, :3]), self.rel[batch.edge_type]], dim=1)
        for _ in range(repeat):
            out = self.query_gnn(graph_size=graph_size, batch=batch)
        pred = torch.log1p(graph_size.float()) + nn.functional.logsigmoid(out)
        return (pred, 0.0) if repeat > 1 else pred


@pytest.fixture
def model():
    torch.manual_seed(0)
    return FakeFICE().eval()


def test_capture_sees_decoder_input_and_reproduces(model):
    batch, gs = make_batch(), torch.tensor([5005])
    with torch.no_grad(), DecoderCapture(model) as cap:
        pred = model(batch, gs)
    assert len(cap.records) == 1
    rec = cap.records[0]
    # what the decoder saw = the batch after the in-place changes
    assert rec.x.shape == (batch.num_nodes, FEAT)
    assert torch.equal(rec.edge_attr, batch.edge_attr)
    assert torch.equal(rec.relation_embeddings(), model.rel[batch.edge_type].detach())
    assert torch.equal(rec.prediction, pred)
    assert rec.num_queries == 5 and torch.equal(rec.y, batch.y)
    report = verify_reproduction(model.query_gnn, cap.records)
    assert report["ok"] and report["max_abs_diff"] < 1e-6


def test_records_are_copies(model):
    batch, gs = make_batch(), torch.tensor([5005])
    with torch.no_grad(), DecoderCapture(model) as cap:
        model(batch, gs)
    before = cap.records[0].edge_attr.clone()
    batch.edge_attr.mul_(0)
    assert torch.equal(cap.records[0].edge_attr, before)


def test_hooks_removed_after_context(model):
    with DecoderCapture(model) as cap:
        pass
    with torch.no_grad():
        model(make_batch(), torch.tensor([5005]))
    assert cap.records == []


def test_repeat_is_deduplicated(model):
    with torch.no_grad(), DecoderCapture(model) as cap:
        pred, _ = model(make_batch(), torch.tensor([5005]), repeat=3)
    assert len(cap.records) == 1 and torch.equal(cap.records[0].prediction, pred)


def test_detects_mismatch(model):
    with torch.no_grad(), DecoderCapture(model) as cap:
        model(make_batch(), torch.tensor([5005]))
    cap.records[0].output = cap.records[0].output + 1.0
    with pytest.raises(AssertionError):
        verify_reproduction(model.query_gnn, cap.records)


def test_split_and_roundtrip(model, tmp_path):
    with torch.no_grad(), DecoderCapture(model) as cap:
        model(make_batch(), torch.tensor([5005]))
    rec = cap.records[0]
    parts = rec.split_queries()
    assert [p["edge_attr"].size(0) for p in parts] == [2, 3, 3, 2, 4]
    assert all(p["edge_index"].min() >= 0 for p in parts)
    single = Batch.from_data_list([Data(x=p["x"], edge_index=p["edge_index"], edge_attr=p["edge_attr"]) for p in parts[1:2]])
    with torch.no_grad():
        out = model.query_gnn(batch=single, graph_size=rec.graph_size)
    assert torch.allclose(out, parts[1]["output"].view(-1), atol=1e-6)

    path = save_records(cap.records, tmp_path / "rec.pt", meta={"seed": 0})
    loaded, meta = load_records(path)
    assert meta == {"seed": 0}
    assert torch.equal(loaded[0].edge_attr, rec.edge_attr)
    assert torch.allclose(replay(model.query_gnn, loaded[0]), rec.output)
