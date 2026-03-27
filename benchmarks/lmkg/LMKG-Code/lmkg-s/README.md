# FICE Comment
The model architecture (`query_graph.py`, `complex_reader.py`, `data_processor_release.py`) is
identical to upstream. The training/eval harness was extended to support multi-dataset / inductive
evaluation and several bugs were fixed.

### `lmkgs.py`

**Bug fixes**
- `Adam` constructor: deprecated `lr=` → `learning_rate=`; added `try/except` fallback for
  `tensorflow.keras.optimizers.legacy.Adam` (required for TF ≥ 2.11).
- Global `MIN`/`MAX` initialised to `0.0` at module level to prevent `NameError` on first eval.
- `write_MIN_MAX`: added `os.makedirs("final_datasets/minmax", exist_ok=True)` so the run no
  longer crashes if the directory does not exist.
- `normalize_MINMAX`: added `if MAX == MIN: return np.zeros_like(y)` to avoid division by zero.
- Scale-3 normalisation: replaced `np.log(y)` with `np.log(np.maximum(y, 1))` to prevent
  `log(0)` crashes on zero-cardinality queries. During training `MIN`/`MAX` are stored in memory
  directly; during eval they are read from file and similarly clamped.

**Multi-dataset / inductive support**
- Unknown dataset names no longer cause a hard exit. Instead `statistics/{dataset}_stats.json`
  is read to obtain `d` (distinct nodes), `b` (distinct predicates), and the available query
  sizes. This enables arbitrary new datasets (`wn18rr_v4`, `fb237_v4`, `dbpedia100k`,
  `wikidata`, …) without modifying the code.
- `n` and `e` are always derived dynamically: `n = query_join + 1`, `e = query_join`. The old
  hardcoded thresholds (`>= 5`, `>= 8`) are removed.
- For `allstar` / `allchain` / `alltypessizes` modes, `n`/`e` are set to the maximum join size
  found in the stats JSON, and the file lists are built from the JSON rather than the hardcoded
  `[2]` list.

**Eval robustness**
- Eval files are discovered with `glob.glob()` pattern matching instead of the hardcoded
  `eval_600_*` prefix; a `FileNotFoundError` is raised when no file matches.
- `combined` eval: `file_name_star`/`file_name_chain` are initialised to `None` and only the
  branch required by `--test-mode` is loaded (avoids loading both files when testing star-only
  or chain-only).
- `allstar`/`alltypessizes` eval uses `eval_star_sizes` from the stats JSON (falls back to
  `star_sizes`) so held-out eval sets can differ from training sets.

### `lmkgs.py` — continuous prediction logging

**Per-epoch prediction logging**
- The single `model.fit(epochs=N)` call is replaced with a per-epoch loop (`model.fit(epochs=1)`
  repeated N times). After each epoch the model predicts on the training set, denormalizes the
  outputs (scale-3: `exp(denormalize_MINMAX(preds))`), and calls `write_predictions_csv()` with
  `stage="train"`, `epoch=epoch_num+1`, `append=True`.
- `query_set_name` is derived from dataset name and base query type
  (e.g. `wn18rr_v4_ind_star_queries`) and propagated to all logging calls.
- `check_network_estimates()` at the end of training is called with `write_csv=False` to avoid
  duplicating the final epoch's rows; for eval it keeps `write_csv=True`.

### `estimates_file.py`
- Added `math.isfinite()` guard to skip non-finite prediction or true-cardinality values.
- Safe average: `average_estimation_error` divided by `wrong_estimates` only when
  `wrong_estimates > 0` (prevents `ZeroDivisionError` when all estimates are correct).

**Updated `write_predictions_csv()`**
- New parameters: `query_set_name`, `epoch=1`, `stage="eval"`, `append=False`.
- Added `"epoch"` column to the CSV fieldnames and each row.
- Per-run file open mode is `"a"` when `append=True` (header written only once on first call),
  `"w"` otherwise.
- Signature of `check_network_estimates()` extended with `query_set_name`, `epoch`, `stage`,
  and `write_csv=True`; all are forwarded to `write_predictions_csv()`.

**Output files**
- `statistics/predictions-{model_name}.csv` — per-run, all epochs × all queries for training;
  single eval pass for eval. Columns: `stage, epoch, dataset_name, query_set_name, query_type,
  batch_idx, prediction, true_value`.
- `statistics/all_predictions.csv` — master file, accumulates across all runs.

# Original README.md
# LMKG-S

LMKG-S is a supervised neural network cardinality estimator for knowledge graphs.


## Quick start

Set up a conda environment with dependencies installed:

```bash
# Install Python environment
conda env create -f environment.yml
conda activate lmkg-s
# Run commands below inside this directory.
```

## Datasets and Queries
To run the model for the SWDF dataset, download everythig from the folder **lmkg-s_data/** from the [dropbox link](https://www.dropbox.com/sh/709rxcpyl631kyk/AAAfbEXXnjQvacrgOPGbg87oa?dl=0) and place inside the folder **final_datasets**.
The queries for the swdf star 2 and chain 2 datasets are located in the same folder and start with the prefix _eval__.

For LUBM and YAGO, you can download the datasets from the [dropbox link](https://www.dropbox.com/sh/709rxcpyl631kyk/AAAfbEXXnjQvacrgOPGbg87oa?dl=0) and create training and evaluation data of preferred size. The python file is already set up for all the different datasets, you just need to specify the dataset name as a parameter when running the model, i.e., --dataset=swdf.

## Running experiments
The training params are:
```bash
python lmkgs.py --training --query-type=["combined","star","chain","complex"] --query-join=<int>
--batch-size=<int> --epochs=<int> --dataset="swdf" --query-join=<int> --layers=<int>
--neurons=<int> --decay
```
The query type corresponds to the model grouping strategy. If combined (default option) is chosen,
 a single model for star and chain queries is created (in this scenario of size 2). If complex is chosen a single model
 is trained on a set of complex queries.

The eval params are:
```bash
python lmkgs.py --eval --query-type=["combined","star","chain","complex"] --query-join=<int>
--dataset="swdf" --query-join=<int> --layers=<int>
--neurons=<int> --decay --test-mode=["all","star","chain",]
```
The test mode corresponds to the queries over which the end statistics are evaluated.
A combined model can answer both star and chain queries, but to test on them separately the test mode has to be specified.
Otherwise, the test is conducted on both star and chain queries.
For complex queries the test mode selects 100 queries from each complex query file.

The training for a model for star and chain queries of size 2 (query-size grouping), can be trained in the following way:
```bash
python lmkgs.py --training --query-type="combined" --query-join=2
```

Evaluating the same model can be done using the following command:
```bash
python lmkgs.py --eval --query-type="combined" --query-join=2
```

For training and evaluating a separate model for star 2 or chain 2 use the following commands:
```bash
#train
python lmkgs.py --training --query-type="star" --query-join=2
python lmkgs.py --training --query-type="chain" --query-join=2

#eval
python lmkgs.py --eval --query-type="star" --query-join=2
python lmkgs.py --eval --query-type="chain" --query-join=2
```

## Metrics & Monitoring

The key metrics to track are
* Cardinality estimation accuracy (Q-error)
    * Separated per ranges, medians, query type and query size
* Training time (seconds)
* Evaluation time


## Custom Data Processor

The current project works with a specific format of queries.
For using LMKG-S for other query forms, the following steps need to be performed.
1. Reader of the custom queries.
2. Conversion from queries to a QueryGraph instance.

    The QueryGraph instance requires
    * n - number of nodes in the knowledge graph
    * e - number of edges in the knowledge graph
    * d - number of nodes in the query
    * b - number of edges in the query

    The QueryGraph receives triples as input where a variable starts with "?". A join is denoted with the usage of the same variable.
    The result of the create_graph method in the QueryGraph instance gives the matrices for the specific query, which need to be concatenated and forwarded to the model.

3. Adding the reader to the lmkgs main method.


<sup>The data_processor_release.py file contains the original star and chain reader. Since the extension of LMKG-S for complex queries, the complex_reader.py and query_graph.py are used. The results for star and chain queries remain the same in using either of the methods. The usage of the latter is advised.</sup>


If you compare with this code, or you use it in your research, please cite:

```
@inproceedings{DBLP:conf/edbt/DavitkovaG022,
  author    = {Angjela Davitkova and
               Damjan Gjurovski and
               Sebastian Michel},
  title     = {{LMKG:} Learned Models for Cardinality Estimation in Knowledge Graphs},
  booktitle = {{EDBT}},
  pages     = {2:169--2:182},
  publisher = {OpenProceedings.org},
  year      = {2022}
}
```
