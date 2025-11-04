git clone https://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:x-oauth-basic@github.com/lukasketzer/fully-inductive-cardinality-estimation.git
git -C /content/fully-inductive-cardinality-estimation pull
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
pip install torch_geometric

pip install -r /content/fully-inductive-cardinality-estimation/requirements.txt