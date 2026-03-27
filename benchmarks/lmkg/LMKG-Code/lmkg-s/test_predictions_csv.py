import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))


def test_write_predictions_csv_per_run_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(tmp_path / "statistics")
    from estimates_file import write_predictions_csv

    preds = [[3.0], [10.0], [1.0]]
    trues = [3, 9, 2]
    write_predictions_csv(preds, trues, "run_A", "swdf", "star")

    csv_path = tmp_path / "statistics" / "predictions-run_A.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["stage"] == "eval"
    assert rows[0]["dataset_name"] == "swdf"
    assert rows[0]["query_type"] == "star"
    assert rows[0]["batch_idx"] == "0"
    assert float(rows[0]["prediction"]) == pytest.approx(3.0)
    assert float(rows[0]["true_value"]) == pytest.approx(3.0)
    assert set(rows[0].keys()) == {"stage", "dataset_name", "query_set_name", "query_type", "batch_idx", "prediction", "true_value"}


def test_write_predictions_csv_master_file_accumulates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(tmp_path / "statistics")
    from estimates_file import write_predictions_csv

    write_predictions_csv([[5.0]], [5], "run_A", "swdf", "star")
    write_predictions_csv([[7.0]], [8], "run_B", "lubm", "path")

    master = tmp_path / "statistics" / "all_predictions.csv"
    assert master.exists()
    with open(master) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["dataset_name"] == "swdf"
    assert rows[1]["dataset_name"] == "lubm"
    assert rows[1]["query_type"] == "path"


import pytest
