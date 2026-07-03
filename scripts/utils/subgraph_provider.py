"""Precalculated, independent per-node neighborhoods on the bipartite/factor graph.

Used when config.neighborhood_mode == 'per_node'. Unlike the joint sampler
(one shared neighborhood around the union of all query seeds in a batch), each
seed node gets its own capped k-hop subgraph, sampled with an RNG seeded by
(node id, sample index). The subgraph — and its cached RWPE — is therefore
identical across batches, epochs, and train/test, eliminating the train/test
neighborhood mismatch.

Monte Carlo sampling: a pool of `pool_size` independent neighborhoods is
precalculated per node. Each forward pass uses `mc_samples` of them (drawn
randomly from the pool during training, the fixed first `mc_samples` at eval)
and the model averages the resulting center embeddings.
"""

import numpy as np
import torch
from typing import Dict, Tuple
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE


def _compute_rwpe(edge_index, edge_attr, num_nodes, walk_length, role_aware):
    """RWPE on one subgraph; mirrors compute_rwpe in scripts/train.py."""
    if walk_length <= 0:
        return None
    transform = AddRandomWalkPE(walk_length=walk_length)
    if role_aware:
        roles = edge_attr[:, 1]
        pe_parts = []
        for role in [1, 2, 3]:  # subject, predicate, object
            mask = (roles.abs() == role)
            sub_data = Data(edge_index=edge_index[:, mask], num_nodes=num_nodes)
            pe_parts.append(transform(sub_data).random_walk_pe)
        return torch.cat(pe_parts, dim=-1)  # (N, 3*walk_length)
    sub_data = Data(edge_index=edge_index, num_nodes=num_nodes)
    return transform(sub_data).random_walk_pe


class PerNodeSubgraphProvider:

    def __init__(self, graph, caps, walk_length=0, role_aware=False, pool_size=1,
                 selection='uniform', ppr_alpha=0.15, degree_features=False):
        """graph: full bipartite/factor graph (Data with edge_index containing
        both directions and edge_attr = [count, role]).

        selection 'uniform': capped k-hop, uniform random neighbors per hop.
        selection 'ppr': capped k-hop, top-cap neighbors per hop by
        personalized PageRank mass (influence-weighted truncation instead of
        uniform truncation); deterministic, so pool_size must be 1.
        selection 'ppr_sample': capped k-hop, neighbors sampled without
        replacement with probability ~ PPR mass (mixed with a uniform floor);
        stochastic and seeded like 'uniform', so it composes with MC pooling.
        """
        self.caps = tuple(int(c) for c in caps)
        self.walk_length = walk_length
        self.role_aware = role_aware
        self.pool_size = pool_size
        self.selection = selection
        self.ppr_alpha = ppr_alpha
        self._ppr_cache = {}
        self._ppr_p = None  # reusable push buffers, lazily allocated
        self._ppr_r = None
        if selection == 'ppr' and pool_size != 1:
            raise ValueError("PPR selection is deterministic; requires pool_size=1")
        self.num_nodes = graph.num_nodes
        self.n_entities = int(graph.original_num_nodes)
        self.n_relations = int(graph.original_num_relations)
        # CSR over the directed edges (both directions are stored in the graph)
        edge_index = graph.edge_index.cpu()
        edge_attr = graph.edge_attr.cpu()
        perm = torch.argsort(edge_index[0], stable=True)
        self.indices = edge_index[1][perm].numpy()
        self.edge_counts = edge_attr[perm, 0].to(torch.float32).numpy()
        self.edge_roles = edge_attr[perm, 1].to(torch.float32).numpy()
        degrees = torch.bincount(edge_index[0], minlength=self.num_nodes)
        self.indptr = torch.cat([torch.zeros(1, dtype=torch.long),
                                 degrees.cumsum(0)]).numpy()
        self.cache: Dict[Tuple[int, int], tuple] = {}
        self._train_rng = np.random.default_rng()
        # True (uncapped) structural features from the full graph
        self._node_role_deg = None  # (num_nodes, 3) log1p degree per |role|
        self._rel_fanout = None     # (num_relations, 2) log1p max out/in fan-out
        if degree_features:
            self._compute_degree_features()

    def _compute_degree_features(self):
        """log1p true degree per role for every node, plus GBJO's per-relation
        max fan-out bounds (max triples sharing one subject / one object).
        One vectorized pass over the full edge list."""
        n = self.num_nodes
        src = np.repeat(np.arange(n, dtype=np.int64),
                        np.diff(self.indptr).astype(np.int64))
        roles = np.abs(self.edge_roles).astype(np.int64)  # 1=subj, 2=pred, 3=obj
        deg = np.zeros((n, 3), dtype=np.float32)
        for k in (1, 2, 3):
            m = roles == k
            if m.any():
                deg[:, k - 1] = np.bincount(src[m], minlength=n).astype(np.float32)
        self._node_role_deg = np.log1p(deg)

        t0 = self.n_entities + self.n_relations
        n_tr = n - t0
        fan = np.zeros((self.n_relations, 2), dtype=np.float32)
        if n_tr > 0 and self.n_relations > 0:
            tmask = src >= t0
            tsrc = src[tmask] - t0
            tdst = self.indices[tmask].astype(np.int64)
            trole = roles[tmask]
            subj = np.zeros(n_tr, dtype=np.int64)
            rel = np.zeros(n_tr, dtype=np.int64)
            obj = np.zeros(n_tr, dtype=np.int64)
            subj[tsrc[trole == 1]] = tdst[trole == 1]
            rel[tsrc[trole == 2]] = tdst[trole == 2] - self.n_entities
            obj[tsrc[trole == 3]] = tdst[trole == 3]
            for col, ent in ((0, subj), (1, obj)):
                keys, cnt = np.unique(ent * self.n_relations + rel,
                                      return_counts=True)
                np.maximum.at(fan[:, col], keys % self.n_relations,
                              cnt.astype(np.float32))
        self._rel_fanout = np.log1p(fan)

    def _degree_feature_rows(self, node_ids):
        """(n, 5) float32: [deg_subj, deg_pred, deg_obj, fan_out, fan_in];
        fan-out columns are nonzero only on relation nodes."""
        nid = node_ids.numpy()
        f = np.zeros((len(nid), 5), dtype=np.float32)
        f[:, :3] = self._node_role_deg[nid]
        is_rel = (nid >= self.n_entities) & (nid < self.n_entities + self.n_relations)
        if is_rel.any():
            f[is_rel, 3:] = self._rel_fanout[nid[is_rel] - self.n_entities]
        return torch.from_numpy(f)

    def _sample_subgraph(self, node, sample_idx):
        """Capped k-hop neighborhood around `node` (local id 0). RNG is seeded
        by (node, sample_idx), so the subgraph is fixed across calls/runs.
        With selection 'ppr_sample', neighbors are drawn without replacement
        with probability ~ PPR mass (10% uniform floor) instead of uniformly."""
        rng = np.random.default_rng((int(node), int(sample_idx)))
        p = self._cached_ppr(node) if self.selection == 'ppr_sample' else None
        indptr, indices = self.indptr, self.indices
        pos = {int(node): 0}
        nodes = [int(node)]
        # (src_local, dst_local, role) -> count; both directions are added for
        # every sampled edge so messages flow both ways on the sampled structure
        edges = {}
        frontier = [int(node)]
        for cap in self.caps:
            nxt = []
            for u in frontier:
                lo, hi = int(indptr[u]), int(indptr[u + 1])
                deg = hi - lo
                if deg <= cap:
                    sel = range(lo, hi)
                elif p is not None:
                    w = p[indices[lo:hi]].astype(np.float64)
                    s = w.sum()
                    w = (0.9 * w / s if s > 0 else 0.0) + 0.1 / deg
                    w /= w.sum()
                    sel = rng.choice(deg, size=cap, replace=False, p=w) + lo
                else:
                    sel = rng.choice(deg, size=cap, replace=False) + lo
                ul = pos[u]
                for e in sel:
                    v = int(indices[e])
                    r = float(self.edge_roles[e])
                    c = float(self.edge_counts[e])
                    vl = pos.get(v)
                    if vl is None:
                        vl = pos[v] = len(nodes)
                        nodes.append(v)
                        nxt.append(v)
                    edges[(ul, vl, r)] = c
                    edges[(vl, ul, -r)] = c
            frontier = nxt
        if edges:
            keys = list(edges.keys())
            edge_index = torch.tensor([[k[0] for k in keys], [k[1] for k in keys]], dtype=torch.long)
            edge_attr = torch.tensor([[edges[k], k[2]] for k in keys], dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)
        return torch.tensor(nodes, dtype=torch.long), edge_index, edge_attr

    def _ppr_run(self, node):
        """Approximate personalized PageRank from `node` (push algorithm) into
        reusable per-provider buffers. Only nodes that ever receive mass are
        scanned/reset, so cost and transient memory scale with the seed's
        neighborhood, not the graph. Returns (p, touched); the caller must
        call _ppr_reset(touched) once done reading p."""
        n = self.num_nodes
        if self._ppr_p is None:
            self._ppr_p = np.zeros(n, dtype=np.float64)
            self._ppr_r = np.zeros(n, dtype=np.float64)
        p, r = self._ppr_p, self._ppr_r
        indptr, indices = self.indptr, self.indices
        r[node] = 1.0
        touched = np.array([int(node)], dtype=np.int64)
        alpha = self.ppr_alpha
        eps = 1e-5
        for _ in range(100):
            # touched is sorted, so `active` matches the ascending order the
            # previous full flatnonzero scan produced — pushes (and their
            # float accumulation order) are bit-identical to the dense version
            deg = (indptr[touched + 1] - indptr[touched]).astype(np.float64)
            active = touched[r[touched] > eps * np.maximum(deg, 1.0)]
            if active.size == 0:
                break
            pushed = []
            for u in active:
                ru = r[u]
                r[u] = 0.0
                p[u] += alpha * ru
                lo, hi = int(indptr[u]), int(indptr[u + 1])
                if hi > lo:
                    np.add.at(r, indices[lo:hi], (1.0 - alpha) * ru / (hi - lo))
                    pushed.append(indices[lo:hi])
                else:
                    p[u] += (1.0 - alpha) * ru  # dangling node keeps its mass
            if pushed:
                touched = np.union1d(touched, np.concatenate(pushed))
        return p, touched

    def _ppr_reset(self, touched):
        self._ppr_p[touched] = 0.0
        self._ppr_r[touched] = 0.0

    def _cached_ppr(self, node):
        """PPR scores per seed as a dense float32 vector, cached. Only used by
        'ppr_sample' (scores are re-read across pool samples); deterministic
        'ppr' reads the buffers directly and never caches."""
        p = self._ppr_cache.get(int(node))
        if p is None:
            pbuf, touched = self._ppr_run(node)
            p = np.zeros(self.num_nodes, dtype=np.float32)
            p[touched] = pbuf[touched].astype(np.float32)
            self._ppr_reset(touched)
            self._ppr_cache[int(node)] = p
        return p

    def _ppr_subgraph(self, node):
        """Capped k-hop expansion around `node` (local id 0), but at each hop
        the top-`cap` neighbors by personalized PageRank mass are kept instead
        of a uniform random sample. Connectivity by construction (same shape
        as the uniform sampler — a pure selection-rule ablation), deterministic.

        Note: a plain induced subgraph over the global top-k PPR nodes does NOT
        work on factor graphs — for hub relation seeds the top-k is dominated
        by 2-hop entities whose connecting triple nodes are not selected,
        yielding an edgeless subgraph."""
        p, touched = self._ppr_run(node)
        indptr, indices = self.indptr, self.indices
        pos = {int(node): 0}
        nodes = [int(node)]
        edges = {}
        frontier = [int(node)]
        for cap in self.caps:
            nxt = []
            for u in frontier:
                lo, hi = int(indptr[u]), int(indptr[u + 1])
                deg = hi - lo
                if deg <= cap:
                    sel = range(lo, hi)
                else:
                    # float32 cast keeps argsort input (and tie behavior)
                    # identical to the previous float32-cached scores
                    sel = lo + np.argsort(p[indices[lo:hi]].astype(np.float32))[::-1][:cap]
                ul = pos[u]
                for e in sel:
                    v = int(indices[e])
                    r = float(self.edge_roles[e])
                    c = float(self.edge_counts[e])
                    vl = pos.get(v)
                    if vl is None:
                        vl = pos[v] = len(nodes)
                        nodes.append(v)
                        nxt.append(v)
                    edges[(ul, vl, r)] = c
                    edges[(vl, ul, -r)] = c
            frontier = nxt
        self._ppr_reset(touched)
        if edges:
            keys = list(edges.keys())
            edge_index = torch.tensor([[k[0] for k in keys], [k[1] for k in keys]], dtype=torch.long)
            edge_attr = torch.tensor([[edges[k], k[2]] for k in keys], dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)
        return torch.tensor(nodes, dtype=torch.long), edge_index, edge_attr

    def _entry(self, node, sample_idx):
        key = (int(node), int(sample_idx))
        e = self.cache.get(key)
        if e is None:
            if self.selection == 'ppr':
                node_ids, edge_index, edge_attr = self._ppr_subgraph(node)
            else:
                node_ids, edge_index, edge_attr = self._sample_subgraph(node, sample_idx)
            pe = _compute_rwpe(edge_index, edge_attr, node_ids.numel(),
                               self.walk_length, self.role_aware)
            # node types (entity=0, relation=1, triple=2) from global id layout
            node_type = torch.zeros(node_ids.numel(), dtype=torch.long)
            node_type[node_ids >= self.n_entities] = 1
            node_type[node_ids >= self.n_entities + self.n_relations] = 2
            feats = (self._degree_feature_rows(node_ids)
                     if self._node_role_deg is not None else None)
            e = self.cache[key] = (edge_index, edge_attr, pe, node_type, feats, node_ids.numel())
        return e

    def draw_sample_ids(self, mc_samples, training):
        """Sample indices used for one forward pass. Random subset of the pool
        during training, the fixed first `mc_samples` at eval."""
        if training and self.pool_size > mc_samples:
            return np.sort(self._train_rng.choice(self.pool_size, size=mc_samples, replace=False))
        return np.arange(mc_samples)

    def build_batch(self, seeds, mc_samples, training):
        """Disjoint batch over per-seed subgraphs.

        seeds: (S,) global bipartite/factor node ids. Returns a Data with
        S * mc_samples subgraphs (ordered seed-major), where `center_index`
        holds the local index of each subgraph's seed node.
        """
        eis, eas, pes, ntypes, feats_l, bvec, centers = [], [], [], [], [], [], []
        off = 0
        gi = 0
        for s in seeds.tolist():
            sample_ids = self.draw_sample_ids(mc_samples, training)
            for k in sample_ids:
                edge_index, edge_attr, pe, node_type, feats, n = self._entry(s, k)
                eis.append(edge_index + off)
                eas.append(edge_attr)
                if pe is not None:
                    pes.append(pe)
                if feats is not None:
                    feats_l.append(feats)
                ntypes.append(node_type)
                bvec.append(torch.full((n,), gi, dtype=torch.long))
                centers.append(off)  # seed is local node 0 of its subgraph
                off += n
                gi += 1
        data = Data(
            edge_index=(torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long)),
            edge_attr=(torch.cat(eas, dim=0) if eas else torch.zeros((0, 2), dtype=torch.float32)),
            num_nodes=off,
        )
        data.node_type = (torch.cat(ntypes) if ntypes else torch.zeros(0, dtype=torch.long))
        data.batch_vec = (torch.cat(bvec) if bvec else torch.zeros(0, dtype=torch.long))
        data.center_index = torch.tensor(centers, dtype=torch.long)
        if pes:
            data.random_walk_pe = torch.cat(pes, dim=0)
        if feats_l:
            data.node_feats = torch.cat(feats_l, dim=0)
        data.per_node = True
        data.mc_samples = mc_samples
        data.num_seeds = seeds.numel()
        return data


_PROVIDER_REGISTRY: Dict[str, PerNodeSubgraphProvider] = {}


def get_provider(graph, config) -> PerNodeSubgraphProvider:
    """Provider per dataset, cached across batches and epochs."""
    provider = _PROVIDER_REGISTRY.get(graph.dataset_name)
    if provider is None:
        provider = PerNodeSubgraphProvider(
            graph,
            caps=config.GNNNeighborhood,
            walk_length=config.random_walk_pe,
            role_aware=(config.relation_gnn.source == 'factor'),
            pool_size=config.mc_pool_size,
            selection=config.neighbor_selection,
            ppr_alpha=config.ppr_alpha,
            degree_features=config.degree_features,
        )
        _PROVIDER_REGISTRY[graph.dataset_name] = provider
    return provider
