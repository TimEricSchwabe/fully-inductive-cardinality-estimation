# Pre-registration (kickoff Sec. 5.5)

Predictions are committed **before** the computation. Deviations are explained under
"Deviations", not silently corrected.

## Assumptions for all 1-WL computations
- Factor graph from `make_factor_graph` (`scripts/utils/data_helper.py`), full graph,
  no neighbour sampling
- Initial colour = node type (entity / relation / triple), matches `relational: True`
- Edge colour = (count, role) from `edge_attr`
- The encoder has 4 layers (`n_embedding_layers: 4`) → the relevant colouring is after 4 rounds
- Computed with `python scripts/wl_colors.py <dataset>`

---

## fano
Generator: `python scripts/generate_controlled.py fano`
Parameters: 7 lines of the Fano plane, 40 subjects per line, fan-out 1 → 840 triples, 63 queries

| | Prediction (reason)                                                                       | Computed        | Match? |
|---|-------------------------------------------------------------------------------------------|-----------------|--------|
| 1-WL unseeded, after 4 rounds | 1 class {p0…p6} (all relations equivalent under the automorphism group of the fano plane) | 1 class {p0…p6} | yes    |
| 1-WL unseeded, stable | 1 class                                                                                   | 1 class         | yes    |
| 1-WL seeded |                                                                                           |                 |        |
| Cardinality structure | 1-star: 120, <br/>2-star: 40, <br/>3-star: 40 (lines) / <br/>0 (non-lines)                               |checked against queries.json                 | yes    |
| Interaction order | 3rd order                                                                                 |                 |        |

Deviations: --

## fano_unequal
Generator: `python scripts/generate_controlled.py fano_unequal --seed 0`
Parameters: Fano plane, block sizes drawn from [20, 80] (seed 0): [71, 58, 51, 36,22, 24], fan-out 1

| | Prediction (reason)                                                                                                                                                                 | Computed                            | Match? |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|--------|
| 1-WL unseeded, after 4 rounds | 7 classes (different cardinalities for each block)                                                                                                                                  | 6 classes {p0},{p1},{p2},{p3},{p4},{p5,p6} | no     |
| 1-WL unseeded, stable | 7 classes                                                                                                                                                                           | 7 classes                           | yes    |
| 1-WL seeded |                                                                                                                                                                                     |                                     |        |
| Cardinality structure | 1-star p: sum of the sizes of the 3 lines through p,<br/>2-star {p,q}: size of the unique line through p and q,<br/>3-star {p,q,r}: size of the block, if {p,q,r} is a line, else 0 | checked against queries.json                                    | yes    |
| Interaction order | 3rd order                                                                                                                                                                           |                                     |        |

Deviations: Predicted 7 classes after 4 rounds, computed 6: p5 and p6 have the same occurrence count (111 = 51+24+36 = 51+38+22) and are separated only via the sizes of their lines which requires more than 4 rounds

## fano_fanout
Generator: `python scripts/generate_controlled.py fano_fanout`
Parameters: Fano plane, 40 subjects per line, fan-out 2 for all predicates → 1680 triples

| | Prediction (reason)                                                                                                                 | Computed | Match? |
|---|-------------------------------------------------------------------------------------------------------------------------------------|----------|--------|
| 1-WL unseeded, after 4 rounds | 1 class {p0...p6} (same as fano)                                                                                                    | 1 class  | yes    |
| 1-WL unseeded, stable | 1 class                                                                                                                             | 1 class  | yes    |
| 1-WL seeded |                                                                                                                                     |          |        |
| Cardinality structure | 1-star: 240, <br/>2-star: 160, <br/>3-star: 320 (lines) / 0 (non-lines); <br/>fanout = 2 (every subject has 2 objects per predicate -> each subject is uncluded 2^k times) |checked against queries.json          | yes    |
| Interaction order | 3rd order                                                                                                                           |          |        |

Deviations: --

## orthogonal
Generator: `python scripts/generate_controlled.py orthogonal`
Parameters: 7 predicates, each on its own 120 subjects, fan-out 1 → 840 triples, 63 queries

| | Prediction (reason)                                    | Computed | Match? |
|---|--------------------------------------------------------|----------|--------|
| 1-WL unseeded, after 4 rounds | 1 class {p0…p6} (all relations structurally identical) | 1 class  | yes    |
| 1-WL unseeded, stable | 1 class                                                | 1 class  | yes    |
| 1-WL seeded |                                                        |          |        |
| Cardinality structure | order 1 = 120, <br/>order ≥ 2 = 0                           |checked against queries.json          | yes    |
| Interaction order | 2nd order                                              |          |        |

Deviations: --

## two_clusters
Generator: `python scripts/generate_controlled.py two_clusters`
Parameters: clusters {p0,p1,p2} and {p3,p4,p5,p6}, 120 subjects each, fan-out 1 → 840 triples, 63 queries

| | Prediction (reason)                                           | Computed                       | Match? |
|---|---------------------------------------------------------------|--------------------------------|--------|
| 1-WL unseeded, after 4 rounds | 2 classes {p0,p1,p2} / {p3…p6} (subject degree 3 vs. 4)       | 2 classes {p0,p1,p2} / {p3…p6} | yes    |
| 1-WL unseeded, stable | 2 classes                        | 2 classes                      | yes    |
| 1-WL seeded |                                                               |                                |        |
| Cardinality structure | star = 120 if all predicates in the same cluster, <br/>otherwise 0 |checked against queries.json                                | yes    |
| Interaction order | 2nd order                                                     |                                |        |


Deviation from the previous report: there 3 + 3 predicates (clusters 1-WL-equivalent), here 3 + 4.

## implication
Generator: `python scripts/generate_controlled.py implication`
Parameters: p0 ⇒ p1 (40 both, 80 only p1), p2 ⇒ p3 (100 both, 20 only p3),
chain p4 ⇒ p5 ⇒ p6 (40 each) → counts 40, 120, 100, 120, 40, 80, 120; 620 triples

| | Prediction (reason)                                                                                                                                        | Computed  | Match? |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|--------|
| 1-WL unseeded, after 4 rounds | 7 classes (different structure of each predicate)                                                                                                          | 7 classes | yes    |
| 1-WL unseeded, stable | 7 classes                                                                                                                                                  | 7 classes | yes    |
| 1-WL seeded |                                                                                                                                                            |           |        |
| Cardinality structure | 1-star: <br/>40,120,100,120,40,80,120<br/>2-star:{p0,p1}=40, {p2,p3}=100, {p4,p5}=40, {p4,p6}=40, {p5,p6}=80, others=0<br/>3-star: {p4,p5,p6}=40, others=0 | checked against queries.json       | yes    |
| Interaction order | 2nd order                                                                                                                                                  |           |        |

Deviations: --

## ising
Generator: `python scripts/generate_controlled.py ising --seed 0`
Parameters: 10 predicates, clusters {p0,p1,p2} and {p3…p6} with J_in = 1.5, p7–p9 isolated,
h = logit(0.3), 12,000 subjects, fan-out 1 → 84,859 triples, 175 queries

| | Prediction (reason)                                                                                                                                                                                                                                                                                                                      | Computed   | Match? |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------|
| 1-WL unseeded, after 4 rounds | 10 classes (all predicates slightly different cardinalities because of randomness of triple generation)                                                                                                                                                                                                                                  | 10 classes | yes    |
| 1-WL unseeded, stable | 10 classes                                                                                                                                                                                                                                                                                                                               | 10 classes | yes    |
| 1-WL seeded |                                                                                                                                                                                                                                                                                                                                          |            |        |
| Cardinality structure | star cardinality = number of subjects carrying all its predices, expected value N * P(all active) under the Ising model, observed value binomial around it; no zeros (smalles expected value ≈ 324); within-cluster stars above the independence estimate and approximately equal per order; corss-cluster stars ≈ independence estimate |checked against queries.json            | yes    |
| Interaction order | 2nd order                                                                                                                                                                                                                                                                                                                                |            |        |

Deviations: --

## path_dataset
Generator: `cd data/path_dataset/raw && python path_dataset.py`,
then from the repo root `python "data/path_queries/raw/path queries.py"`
Parameters: 150 chains p_1 → p_2 → p_3 → p_4 → p_5, 750 triples, 5 queries (prefixes)

| | Prediction (reason)                                                                     | Computed                       | Match? |
|---|-----------------------------------------------------------------------------------------|--------------------------------|--------|
| 1-WL unseeded, after 4 rounds | 3 classes {p1,p5} (most outer nodes), {p2,p4} (next to outer nodes), {p3} (middle node) | 3 classes {p1},{p2,p3,p4},{p5} | no     |
| 1-WL unseeded, stable | 3 classes                                                                               | 5 classes                      | no     |
| 1-WL seeded |                                                                                         |                                |        |
| Cardinality structure | each query: 150                                                                         |checked against queries.json                                | yes    |
| Interaction order | n/a                                                                                     |                                |        |

Deviations: Prediction assumed the chain looks the same from both ends (p1 = p5). This is wrong: first node of chain is only a subject, last only an object, so p1 ≠ p5. The predicates p2,p3 and p4 are seperated only after round 4.

## cycle_dataset
Generator: `cd data/cycle_dataset/raw && python cycle_dataset.py`,
queries: `cd data/cycle_queries/raw && python "cycle queries.py"`
Parameters: disjoint cycles of length 3–7 (385, 308, 231, 154, 77), one predicate
friendsWith, 5005 triples, 50 queries

| | Prediction (reason)                                                                       | Computed | Match? |
|---|-------------------------------------------------------------------------------------------|----------|--------|
| 1-WL unseeded, after 4 rounds | 1 class (all same structure)                                                              | 1 class  | yes    |
| 1-WL unseeded, stable | 1 class                                                                                   | 1 class  | yes    |
| 1-WL seeded |                                                                                           |          |        |
| Cardinality structure | chains and stars = 5005; cycle L = sum over l dividing L of l · #cycles_l; mismatched = 0 |checked against queries.json          | yes    |
| Interaction order | n/a (single predicate)                                                                    |          |        |

Deviations: --