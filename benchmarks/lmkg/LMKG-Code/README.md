# FICE Comment
LMKG implementation from https://github.com/DamjanGjurovski/LMKG-Code/tree/master
Commit: https://github.com/DamjanGjurovski/LMKG-Code/commit/670bb7edcaf930e92729240cbe8f1a1d7cf14aee (sha1: 670bb7e)

We only used LMKG-S. The model architecture is unchanged from upstream. The surrounding infrastructure
was extended to support multi-dataset / inductive training and evaluation, and several bugs were fixed.
See `lmkg-s/README.md` for the full list of changes.





# Original README.MD
# LMKG: Learned Models for Cardinality Estimation in Knowledge Graphs

A code repository for the paper: _LMKG: Learned Models for Cardinality Estimation in Knowledge Graphs_.

For the code for LMKG-U go to the **lmkg-u/** folder.

For the code for LMKG-S go to the **lmkg-s/** folder.

To download the data and run the models, follow the instructions in the README.md files in lmkg-s and lmkg-u.


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