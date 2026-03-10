# Auto Research

Your goal is to autonomously improve the FICE Cardinality Estimation model that aims to be fully inductive, i.e. generalize to new graphs zero-shot

## Background

The Fice model is a encoder-decoder architecture. One (or 2) GNNS create embeddings from the Knowledge Graph (either the raw KG for entity and the factor graph for relations or getting both embedding types from the factor graph with one gnn). The idea is that the embeddings encode the local neighborhood of the nodes/relations and how often a node/relation cooccurs e.g. with another one, and also higher order motifs, e.g. how often do n node and relations in a certain shape like star,path etc.

Then the embeddings are added in the query graph, which is a graph representation of a BGP. The final query gnn should decode from the query structure plus the embeddings the final log-cardinality

## The experiment loop

Your goal is now to improve on the current baseline (as currently in configs/train_full.yml). You can make smaller changes like change the batch size but the expected improvement is minor. Especially interesting changes are changes on the architecture that are motivated by other GNN/embedding results and are principled based on semantics of cardinality estimation and what the embeddings should encode.

Do the following in a loop:

1. To run a experiment go into the scripts/automated-experiments folder and run source .venv/bin/activate and then python run_experiment.py. This will push the new changes to git, connect to the compute cluster, pull the changes and run the experiment. it then polls every 30s slurm and wandb and when the experiment is finished, will return the lowest metrics.

2. The tracked metrics are overall loss, median and mean q-error and those metrics for the 2 inductive datasets dbpedia100k and wn18 inductive. The goal is ideally to improve an all of those, but the overall most important are the 3 median q-errors

3. Based on the results of the executed run update experiment_logbook.md in the autoresearch folder by adding an entry detailing the changes of the experiment and then the results

4. Based on the performed experiments and principled promising changes make changes to the model architecture and the training flow and continue the experiment loop without asking, always keep going, indefinitely.

5. If there seems to be an error on the server run, you can get the log via: ssh coma "cat fully-inductive-cardinality-estimation/slurm-<job_id>.out"

## Rules (IMPORTANT)

1. You are on the autoresearch branch (and the cluster also). Never change this.
2. never change the evaluation of the model, this needs to stay fixed.
3. never change the execution code for automated research on the cluster.
4. Never change the number of queries and the epochs (as per the configs); they are configured so each experiment runs in about X minutes, so we can iterate fast.
5. NEVER change the used datasets and specific combination of them. This must always be fixed.

## Learnings from previous Experiments

1. Never use the log-selectivity mode alsways log-cardinality. log-selectivity is orders of magnitude worse.
2. factor-factor is similar in performance to KG-factor or KG-bipartite, its a bit better on star queries and a bit worse on path queries