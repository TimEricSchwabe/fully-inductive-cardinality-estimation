# LSS Benchmark — FICE Integration Notes

This directory integrates the [LSS (Learned Sketch for Subgraph Counting)](https://github.com/IntelligentDDS/LSS)
model and [ProNE](https://github.com/THUDM/ProNE) embeddings into the FICE benchmark suite.
The `LSS/` and `ProNE/` subdirectories are unmodified third-party code. All FICE-specific
logic lives in the files listed below.

---

## Files added for FICE integration

| File | Purpose |
|---|---|
| `run_all_experiments.py` | Main entry point; orchestrates all 4 pipeline phases for every experiment in `experiments_config.yaml` |
| `experiments_config.yaml` | LOCO experiment definitions: datasets, query paths, train pool / left-out splits |
| `transform_graph.py` | Converts FICE RDF TTL files to G-Care/LSS graph format; writes `id_to_id_mapping*.json` |
| `transform_query.py` | Legacy/placeholder; query transformation is handled inline in `run_all_experiments.py` |
| `rdf_types.py` | Shared RDF type URI constants used during graph transformation |
| `get_embeddings.py` | Placeholder; embedding generation is handled inline in `run_all_experiments.py` |
| `requirements.txt` | Python dependencies for the benchmark |
| `run_all_experiments.sbatch` | SLURM batch script for running on the LSS cluster |
| `CLAUDE.md` | Guidance for Claude Code when working in this directory |

---

## Modifications to third-party code

### `LSS/active_train.py` — test mode: save `predictions.csv`

**Original behaviour:** `test` mode called `active_learner.evaluate()` but discarded the
return value; no output was written to disk.

**Change:** Capture the return value and write `{save_res_dir}/predictions.csv` in the
standard FICE format:

```
query_id,ground_truth,prediction
0,12345,9871
...
```

Cardinalities are converted from LSS's internal log2 representation back to integer counts
via `round(2 ** value)`.

```diff
 elif args.mode == "test":
+    import csv, math
+    from util import make_dir
     print("loading model ...")
-    load_model(args = args, model=model, device=args.device, optimizer= optimizer)
+    load_model(args=args, model=model, device=args.device, optimizer=optimizer)
     print("make prediction ...")
-    active_learner.evaluate(model=model, criterion= criterion, eval_datasets=val_datasets, print_res=True)
+    all_eval_res = active_learner.evaluate(model=model, criterion=criterion, eval_datasets=val_datasets, print_res=True)
+    make_dir(args.save_res_dir)
+    pred_path = os.path.join(args.save_res_dir, "predictions.csv")
+    with open(pred_path, "w", newline="") as f:
+        writer = csv.writer(f)
+        writer.writerow(["query_id", "ground_truth", "prediction"])
+        qid = 0
+        for res, _loss, _l1, _elapsed in all_eval_res:
+            for card, output in res:
+                writer.writerow([qid, round(math.pow(2, card)), round(math.pow(2, output))])
+                qid += 1
+    print(f"Predictions saved to {pred_path}")
```
