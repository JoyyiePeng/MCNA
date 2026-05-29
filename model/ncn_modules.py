"""
Multi-hop Common Neighbor (CN) Computation Module
"""

import os
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from scipy.sparse import csr_matrix
import math


# ============ Global Cache Management ============
_MULTIHOP_CN_CACHE = {}  # Multi-hop cache

# Bump this whenever the CN computation semantics change so that old caches
# from previous algorithms (e.g. walk-count A^k) are not silently reused.
_CN_ALGO_VERSION = "v2-shell-setintersect"


def _get_multihop_hash(graph, max_hops, cn_threshold, top_m, cn_semantic):
    """Compute unique identifier for multi-hop graph"""
    src, dst = graph.edges()
    edge_str = (
        f"{_CN_ALGO_VERSION}_"
        f"{graph.num_nodes()}_{graph.num_edges()}_"
        f"{src.sum().item():.0f}_{dst.sum().item():.0f}_"
        f"hops{max_hops}_th{cn_threshold}_topm{top_m}_sem{cn_semantic}"
    )
    return hashlib.md5(edge_str.encode()).hexdigest()[:12]


# ============================================================
# Multi-hop CN Computer
# ============================================================

class MultiHopCNComputer:
    """
    Multi-hop Common Neighbor Pre-computer (shell + set-intersection semantics).

    For each hop k, defines the k-hop shell neighborhood
        N^k(v) = { u : shortest-path-distance(v, u) == k }
    and the k-hop common-neighbor count
        C^k(i, j) = | N^k(i) intersect N^k(j) |.

    Implementation outline (per hop, both dense and sparse paths):
      1. Frontier expansion: F = B^{k-1} @ A  (walk counts via the previous
         shell into one more step).
      2. Shell mask: zero out columns already visited in hops 0..k-1, so only
         genuinely-new k-hop nodes survive.
      3. Per-row Top-M truncation on F (using the walk count as a relevance
         tie-breaker), then binarise to obtain B^k -- the indicator matrix
         of N^k.
      4. Set-intersection CN: C^k = B^k @ B^k.T  (diagonal cleared).
      5. Per-row Top-M + threshold on C^k, log1p, row-wise softmax => P^k.

    Truncation happens at every step, so no dense N x N intermediate is ever
    materialised in the sparse path.
    """

    def __init__(self,
                 max_hops: int = 3,
                 cn_threshold: int = 1,
                 top_m: int = 50,
                 cn_semantic: str = "shell_set",
                 max_nodes_dense: int = 10000,
                 cache_dir: str = ".cn_cache",
                 use_file_cache: bool = True):
        if cn_semantic not in {"shell_set", "shell_walk"}:
            raise ValueError(
                f"cn_semantic must be 'shell_set' or 'shell_walk', got {cn_semantic}"
            )
        self.max_hops = max_hops
        self.cn_threshold = cn_threshold
        self.top_m = top_m
        self.cn_semantic = cn_semantic
        self.max_nodes_dense = max_nodes_dense
        self.cache_dir = cache_dir
        self.use_file_cache = use_file_cache

        # Cache data
        self._cn_data_list = None  # list of hop data
        self._is_precomputed = False
        self._use_dense = False
        self._cache_device = None

    def precompute(self, graph, device):
        """Precompute CN matrices for all hops"""
        if self._is_precomputed:
            self._ensure_device(device)
            return

        graph_hash = _get_multihop_hash(
            graph, self.max_hops, self.cn_threshold, self.top_m, self.cn_semantic
        )

        # 1. Check memory cache
        if graph_hash in _MULTIHOP_CN_CACHE:
            print(f"[MultiHop CN] Loading from memory: {graph_hash}")
            self._load_from_cache(_MULTIHOP_CN_CACHE[graph_hash], device)
            return

        # 2. Check file cache
        if self.use_file_cache:
            cache_path = os.path.join(self.cache_dir, f"multihop_cn_{graph_hash}.pt")
            os.makedirs(self.cache_dir, exist_ok=True)
            if os.path.exists(cache_path):
                print(f"[MultiHop CN] Loading from file: {cache_path}")
                # weights_only=False is required because the cache stores
                # DGLGraph objects (sparse-path) which torch.load 2.6+ blocks
                # by default. Cache files come from this code, so trusting the
                # pickle is fine.
                cn_data = torch.load(cache_path, map_location='cpu', weights_only=False)
                _MULTIHOP_CN_CACHE[graph_hash] = cn_data
                self._load_from_cache(cn_data, device)
                return

        # 3. Compute
        print(f"[MultiHop CN] Computing... (nodes={graph.num_nodes()}, hops={self.max_hops})")
        cn_data = self._compute_all_hops(graph, device)

        # 4. Save
        _MULTIHOP_CN_CACHE[graph_hash] = cn_data
        if self.use_file_cache:
            cache_path = os.path.join(self.cache_dir, f"multihop_cn_{graph_hash}.pt")
            torch.save(self._to_cpu(cn_data), cache_path)
            print(f"[MultiHop CN] Saved: {cache_path}")

        self._load_from_cache(cn_data, device)

    def _compute_all_hops(self, graph, device):
        """Compute CN for all hops"""
        num_nodes = graph.num_nodes()
        use_dense = num_nodes <= self.max_nodes_dense

        if graph.num_edges() == 0:
            return {'empty': True, 'use_dense': use_dense, 'hop_data': []}

        if use_dense:
            return self._compute_dense(graph, device, num_nodes)
        else:
            return self._compute_sparse(graph, device, num_nodes)

    def _compute_dense(self, graph, device, num_nodes):
        """Dense path. Two semantic modes:

        - shell_set  : C^k = B^k @ B^k.T, where B^k is the binary indicator of
                       the exactly-k-hop shell N^k(v). Counts unique common
                       k-hop neighbours.
        - shell_walk : C^k = A^k masked to the exactly-k-hop shell positions
                       (entries where shortest-distance equals k). Counts the
                       multiplicity of length-k walks restricted to k-hop
                       pairs -- preserves the numerical gradient that the
                       binary set-intersection loses.
        """
        # Build the adjacency directly from the edge list to avoid a hard
        # dependency on dgl.heterograph.adjacency_matrix(), which can be broken
        # by mismatched dgl/torch sparse-module builds in some environments.
        src, dst = graph.edges()
        adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.float32)
        adj[src.long(), dst.long()] = 1.0
        adj.fill_diagonal_(0)  # safety: ignore any self-loops in the input

        top_m = min(self.top_m, num_nodes - 1)
        eye = torch.eye(num_nodes, device=device)
        visited = eye.clone()     # already-reached set, includes self

        hop_data = []

        if self.cn_semantic == "shell_set":
            b_prev = eye.clone()  # B^0 = I  (each node "is at" itself)

            for hop in range(1, self.max_hops + 1):
                # --- Step 1: frontier in walk-count form (B^{k-1} @ A) ---
                frontier = torch.mm(b_prev, adj)

                # --- Step 2: shell mask (drop already-visited nodes) ---
                shell_counts = frontier * (visited == 0).float()

                # --- Step 3: per-row Top-M on the walk counts ---
                if shell_counts.numel() > 0 and top_m > 0:
                    top_values, top_indices = torch.topk(shell_counts, k=top_m, dim=1)
                    shell_counts_topm = torch.zeros_like(shell_counts)
                    shell_counts_topm.scatter_(1, top_indices, top_values)
                    shell_counts = shell_counts_topm

                # --- Step 4: binarise to B^k ---
                b_k = (shell_counts > 0).float()

                # Update visited |= B^k for the next iteration.
                visited = ((visited + b_k) > 0).float()

                # --- Step 5: true CN via set intersection ---
                c_k = torch.mm(b_k, b_k.t())
                c_k.fill_diagonal_(0)

                if self.cn_threshold > 1:
                    c_k = c_k * (c_k >= self.cn_threshold).float()

                # --- Step 6: per-row Top-M on C^k ---
                top_values, top_indices = torch.topk(c_k, k=top_m, dim=1)
                c_k_topm = torch.zeros_like(c_k)
                c_k_topm.scatter_(1, top_indices, top_values)
                c_k = c_k_topm

                attn_matrix, has_cn = self._cn_to_attention_dense(c_k)
                hop_data.append({'attn_matrix': attn_matrix, 'has_cn': has_cn})
                print(
                    f"  [Hop-{hop}] shell-set: shell avg "
                    f"{b_k.sum(dim=1).mean().item():.1f}, "
                    f"valid CN nodes {int(has_cn.sum().item())}/{num_nodes}"
                )
                b_prev = b_k

        else:  # shell_walk
            # Maintain full walk count W^k (no truncation) so that downstream
            # shell masking remains exact at higher hops.
            walk_count = eye.clone()  # W^0 = I

            for hop in range(1, self.max_hops + 1):
                # --- Step 1: walk count W^k = W^{k-1} @ A ---
                walk_count = torch.mm(walk_count, adj)

                # --- Step 2: shell mask -- keep only positions never reached ---
                shell_walk = walk_count * (visited == 0).float()

                # --- Update visited from full shell, before truncation ---
                visited = ((visited + (shell_walk > 0).float()) > 0).float()

                # --- Step 3: per-row Top-M (this becomes C^k directly) ---
                top_values, top_indices = torch.topk(shell_walk, k=top_m, dim=1)
                c_k = torch.zeros_like(shell_walk)
                c_k.scatter_(1, top_indices, top_values)

                if self.cn_threshold > 1:
                    c_k = c_k * (c_k >= self.cn_threshold).float()

                attn_matrix, has_cn = self._cn_to_attention_dense(c_k)
                hop_data.append({'attn_matrix': attn_matrix, 'has_cn': has_cn})
                print(
                    f"  [Hop-{hop}] shell-walk: shell nnz "
                    f"{int((shell_walk > 0).sum().item())}, "
                    f"avg walk count {shell_walk[shell_walk > 0].mean().item() if (shell_walk > 0).any() else 0.0:.2f}, "
                    f"valid CN nodes {int(has_cn.sum().item())}/{num_nodes}"
                )

        return {'empty': False, 'use_dense': True, 'hop_data': hop_data}

    @staticmethod
    def _cn_to_attention_dense(c_k):
        """Convert a [N, N] CN-style matrix to (attn, has_cn) via log1p+softmax."""
        has_cn = (c_k.sum(dim=1, keepdim=True) > 0).float()
        if has_cn.sum() > 0:
            attn_scores = torch.log1p(c_k)
            attn_scores = attn_scores.masked_fill(c_k == 0, -1e9)
            attn_matrix = F.softmax(attn_scores, dim=1)
            attn_matrix[has_cn.squeeze(-1) == 0] = 0
        else:
            attn_matrix = torch.zeros_like(c_k)
        return attn_matrix, has_cn

    @staticmethod
    def _topk_per_row_csr(mat, top_m):
        """Keep at most `top_m` largest entries per row of a scipy csr_matrix.

        Used twice per hop: once to truncate the shell frontier (relevance
        ranked by walk count) and once to truncate C^k (ranked by CN count).
        """
        csr = mat.tocsr()
        n_rows = csr.shape[0]
        new_data = []
        new_indices = []
        new_indptr = np.zeros(n_rows + 1, dtype=np.int64)

        for i in range(n_rows):
            row_start, row_end = csr.indptr[i], csr.indptr[i + 1]
            row_data = csr.data[row_start:row_end]
            row_idx = csr.indices[row_start:row_end]
            if row_data.size <= top_m:
                new_data.append(row_data)
                new_indices.append(row_idx)
            else:
                # argpartition gives the indices of the top_m largest values.
                top_k_idx = np.argpartition(row_data, -top_m)[-top_m:]
                new_data.append(row_data[top_k_idx])
                new_indices.append(row_idx[top_k_idx])
            new_indptr[i + 1] = new_indptr[i] + new_data[-1].size

        if new_data:
            new_data = np.concatenate(new_data)
            new_indices = np.concatenate(new_indices)
        else:
            new_data = np.array([], dtype=csr.dtype)
            new_indices = np.array([], dtype=np.int64)

        return csr_matrix(
            (new_data, new_indices, new_indptr),
            shape=mat.shape,
        )

    def _compute_sparse(self, graph, device, num_nodes):
        """Sparse path. Two semantic modes mirror the dense path:

        - shell_set  : C^k = B^k @ B^k.T (set-intersection of k-hop shells).
        - shell_walk : C^k = (W^k masked to k-hop shell positions),
                       preserving the walk-count numerical signal.
        """
        src, dst = graph.edges()
        src_np, dst_np = src.cpu().numpy(), dst.cpu().numpy()

        data = np.ones(len(src_np), dtype=np.float32)
        adj = csr_matrix(
            (data, (src_np, dst_np)),
            shape=(num_nodes, num_nodes),
        )
        # safety: drop any explicit self-loops from the input adjacency
        adj.setdiag(0)
        adj.eliminate_zeros()

        top_m = self.top_m
        idx = np.arange(num_nodes, dtype=np.int64)
        ones = np.ones(num_nodes, dtype=np.float32)
        identity = csr_matrix((ones, (idx, idx)), shape=(num_nodes, num_nodes))
        visited = identity.copy()

        hop_data = []

        if self.cn_semantic == "shell_set":
            b_prev = identity.copy()
            for hop in range(1, self.max_hops + 1):
                frontier = b_prev @ adj
                shell_counts = frontier - frontier.multiply(visited)
                shell_counts.eliminate_zeros()
                shell_counts = self._topk_per_row_csr(shell_counts, top_m)

                b_k = shell_counts.copy()
                if b_k.nnz > 0:
                    b_k.data = np.ones_like(b_k.data)

                visited = visited + b_k
                if visited.nnz > 0:
                    visited.data = np.minimum(visited.data, 1.0)
                visited.eliminate_zeros()

                c_k = b_k @ b_k.T
                c_k.setdiag(0)
                c_k.eliminate_zeros()

                if self.cn_threshold > 1 and c_k.nnz > 0:
                    c_k.data[c_k.data < self.cn_threshold] = 0
                    c_k.eliminate_zeros()

                c_k = self._topk_per_row_csr(c_k, top_m)

                hop_data.append(
                    self._cn_to_attention_sparse(c_k, num_nodes, device)
                )
                if hop_data[-1]['cn_graph'] is None:
                    print(f"  [Hop-{hop}] No valid CN")
                else:
                    print(
                        f"  [Hop-{hop}] shell-set: shell nnz {b_k.nnz}, "
                        f"CN edges {hop_data[-1]['cn_graph'].num_edges()}"
                    )
                b_prev = b_k

        else:  # shell_walk
            walk_count = identity.copy()  # W^0 = I
            for hop in range(1, self.max_hops + 1):
                walk_count = walk_count @ adj  # W^k
                shell_walk = walk_count - walk_count.multiply(visited)
                shell_walk.eliminate_zeros()

                # update visited from full shell, before truncation
                if shell_walk.nnz > 0:
                    new_visited = shell_walk.copy()
                    new_visited.data = np.ones_like(new_visited.data)
                    visited = visited + new_visited
                    visited.data = np.minimum(visited.data, 1.0)
                    visited.eliminate_zeros()

                c_k = self._topk_per_row_csr(shell_walk, top_m)

                if self.cn_threshold > 1 and c_k.nnz > 0:
                    c_k.data[c_k.data < self.cn_threshold] = 0
                    c_k.eliminate_zeros()

                hop_data.append(
                    self._cn_to_attention_sparse(c_k, num_nodes, device)
                )
                if hop_data[-1]['cn_graph'] is None:
                    print(f"  [Hop-{hop}] No valid CN")
                else:
                    avg_walk = float(shell_walk.data.mean()) if shell_walk.nnz > 0 else 0.0
                    print(
                        f"  [Hop-{hop}] shell-walk: shell nnz {shell_walk.nnz}, "
                        f"avg walk count {avg_walk:.2f}, "
                        f"CN edges {hop_data[-1]['cn_graph'].num_edges()}"
                    )

        return {'empty': False, 'use_dense': False, 'hop_data': hop_data}

    @staticmethod
    def _cn_to_attention_sparse(c_k, num_nodes, device):
        """Convert a sparse C^k matrix into the (cn_graph, attn_weights, has_cn)
        record consumed by `aggregate`'s sparse path."""
        c_coo = c_k.tocoo()
        if c_coo.nnz == 0:
            return {
                'cn_graph': None,
                'attn_weights': None,
                'has_cn': torch.zeros(num_nodes, 1, device=device),
            }
        cn_graph = dgl.graph(
            (c_coo.row, c_coo.col), num_nodes=num_nodes, device=device,
        )
        cn_weights = torch.tensor(c_coo.data, dtype=torch.float32, device=device)
        scores = torch.log1p(cn_weights)
        attn_weights = dgl.ops.edge_softmax(cn_graph, scores)
        has_cn = (cn_graph.in_degrees() > 0).float().unsqueeze(1)
        return {
            'cn_graph': cn_graph,
            'attn_weights': attn_weights,
            'has_cn': has_cn,
        }

    def _load_from_cache(self, cn_data, device):
        """Load from cache"""
        if cn_data.get('empty', True):
            self._cn_data_list = []
            self._is_precomputed = True
            return

        self._use_dense = cn_data['use_dense']
        self._cn_data_list = []

        for hop_info in cn_data['hop_data']:
            if self._use_dense:
                self._cn_data_list.append({
                    'attn_matrix': hop_info['attn_matrix'].to(device),
                    'has_cn': hop_info['has_cn'].to(device),
                })
            else:
                if hop_info['cn_graph'] is not None:
                    self._cn_data_list.append({
                        'cn_graph': hop_info['cn_graph'].to(device),
                        'attn_weights': hop_info['attn_weights'].to(device),
                        'has_cn': hop_info['has_cn'].to(device),
                    })
                else:
                    self._cn_data_list.append({
                        'cn_graph': None,
                        'attn_weights': None,
                        'has_cn': hop_info['has_cn'].to(device),
                    })

        self._cache_device = device
        self._is_precomputed = True

    def _to_cpu(self, cn_data):
        """Move to CPU"""
        if cn_data.get('empty', True):
            return cn_data

        hop_data_cpu = []
        for hop_info in cn_data['hop_data']:
            if cn_data['use_dense']:
                hop_data_cpu.append({
                    'attn_matrix': hop_info['attn_matrix'].cpu(),
                    'has_cn': hop_info['has_cn'].cpu(),
                })
            else:
                if hop_info['cn_graph'] is not None:
                    hop_data_cpu.append({
                        'cn_graph': hop_info['cn_graph'].to('cpu'),
                        'attn_weights': hop_info['attn_weights'].cpu(),
                        'has_cn': hop_info['has_cn'].cpu(),
                    })
                else:
                    hop_data_cpu.append({
                        'cn_graph': None,
                        'attn_weights': None,
                        'has_cn': hop_info['has_cn'].cpu(),
                    })

        return {'empty': False, 'use_dense': cn_data['use_dense'], 'hop_data': hop_data_cpu}

    def _ensure_device(self, device):
        """Ensure correct device"""
        if self._cache_device == device:
            return

        for i, hop_info in enumerate(self._cn_data_list):
            if self._use_dense:
                self._cn_data_list[i] = {
                    'attn_matrix': hop_info['attn_matrix'].to(device),
                    'has_cn': hop_info['has_cn'].to(device),
                }
            else:
                if hop_info['cn_graph'] is not None:
                    self._cn_data_list[i] = {
                        'cn_graph': hop_info['cn_graph'].to(device),
                        'attn_weights': hop_info['attn_weights'].to(device),
                        'has_cn': hop_info['has_cn'].to(device),
                    }
        self._cache_device = device

    def aggregate(self, feat, hop_idx, transform_fn=None):
        """
        Aggregate features using CN matrix of specified hop

        Args:
            feat: [N, d] node features
            hop_idx: hop index (0=1-hop, 1=2-hop, 2=3-hop)
            transform_fn: feature transformation function
        """
        h = transform_fn(feat) if transform_fn else feat

        if hop_idx >= len(self._cn_data_list):
            return h

        hop_data = self._cn_data_list[hop_idx]

        if self._use_dense:
            attn_matrix = hop_data['attn_matrix']
            has_cn = hop_data['has_cn']

            out = torch.mm(attn_matrix, h)
            out = has_cn * out + (1 - has_cn) * h
            return out
        else:
            cn_graph = hop_data['cn_graph']
            has_cn = hop_data['has_cn']

            if cn_graph is None:
                return h

            with cn_graph.local_scope():
                cn_graph.ndata['h'] = h
                cn_graph.edata['attn'] = hop_data['attn_weights']
                cn_graph.update_all(
                    fn.u_mul_e('h', 'attn', 'm'),
                    fn.sum('m', 'out')
                )
                out = cn_graph.ndata['out']
                out = has_cn * out + (1 - has_cn) * h
            return out

    @property
    def num_hops(self):
        return len(self._cn_data_list) if self._cn_data_list else 0


# ============================================================
# Gated Fusion Module
# ============================================================

class GatedFusion(nn.Module):
    """Node-level gated fusion with range constraints"""
    def __init__(self, hidden_dim, dropout=0.1,
                 min_gcn_weight=0.1, max_gcn_weight=1, gcn_init_weight=0.5):
        super().__init__()
        # GCN weight range [min, max], MoE automatically [1-max, 1-min]
        self.min_gcn_weight = min_gcn_weight
        self.max_gcn_weight = max_gcn_weight
        self.temperature = nn.Parameter(torch.ones(1))
        self.gcn_init_weight = gcn_init_weight
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Output only 1 value
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gate[-1].weight, std=0.01)
        # Make sigmoid output close to gcn_init_weight position in [min, max] range
        init_ratio = (self.gcn_init_weight - self.min_gcn_weight) / (self.max_gcn_weight - self.min_gcn_weight + 1e-8)
        init_ratio = max(0.01, min(0.99, init_ratio))  # Avoid extreme values
        init_bias = math.log(init_ratio / (1 - init_ratio + 1e-8))
        self.gate[-1].bias.data = torch.tensor([init_bias])

    def forward(self, gcn_out, moe_out):
        gate_input = torch.cat([gcn_out, moe_out], dim=-1)
        logit = self.gate(gate_input)  # [N, 1]

        # sigmoid output in [0, 1], map to [min_gcn, max_gcn]
        ratio = torch.sigmoid(logit)
        gcn_weight = self.min_gcn_weight + (self.max_gcn_weight - self.min_gcn_weight) * ratio
        moe_weight = 1 - gcn_weight

        out = gcn_weight * gcn_out + moe_weight * moe_out

        weights = torch.cat([gcn_weight, moe_weight], dim=-1)
        return out, weights


# ============================================================
# Sparse Multi-hop MoE Layer
# ============================================================

class SparseMultiHopMoE(nn.Module):
    """
    Sparse Multi-hop MoE Layer

    Architecture:
    - Main path: GCN (always selected, not routed)
    - Enhancement path: 1-hop, 2-hop, 3-hop CN (sparse top-k selection)
    """

    def __init__(self,
                 gcn_layer,
                 in_feats: int,
                 out_feats: int,
                 num_hops: int = 3,
                 top_k: int = 2,
                 dropout: float = 0.5,
                 noise_std: float = 0.0,
                 router_dropout: float = 0.0,
                 load_balance_weight: float = 0.01,
                 cn_threshold: int = 1,
                 cn_top_m: int = 50,
                 cn_semantic: str = "shell_set",
                 max_nodes_dense: int = 10000,
                 router_temperature=1.0,
                 min_temperature: float = 1.0,
                 fixed_hop=None,
                 cache_dir: str = ".cn_cache"):
        super().__init__()

        self.gcn_layer = gcn_layer
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_hops = num_hops
        self.top_k = min(top_k, num_hops)
        self.noise_std = noise_std
        self.router_dropout_p = router_dropout
        self.load_balance_weight = load_balance_weight
        self.router_temperature = nn.Parameter(torch.tensor(router_temperature))
        self.min_temperature = min_temperature
        self.fixed_hop = fixed_hop

        # Multi-hop CN computer
        self.cn_computer = MultiHopCNComputer(
            max_hops=num_hops,
            cn_threshold=cn_threshold,
            top_m=cn_top_m,
            cn_semantic=cn_semantic,
            max_nodes_dense=max_nodes_dense,
            cache_dir=cache_dir,
        )

        # Transform layer for each hop
        self.hop_transforms = nn.ModuleList([
            nn.Linear(in_feats, out_feats) for _ in range(num_hops)
        ])

        # Router: looks at GCN output + each hop output
        self.router = nn.Sequential(
            nn.Linear(out_feats * (num_hops + 1), out_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feats, num_hops)
        )

        # Overall enhancement coefficient
        self.alpha = nn.Parameter(torch.zeros(1))

        # Statistics
        self.aux_loss = 0.0
        self._last_expert_stats = {}

        self.input_proj = nn.Linear(in_feats, out_feats) if in_feats != out_feats else nn.Identity()

        # Gated fusion layer
        self.fusion = GatedFusion(out_feats, dropout=0.3)
        self.layer_norm = nn.LayerNorm(out_feats)

    def forward(self, graph, feat, main_out=None):
        device = feat.device

        h_original = feat
        # Precompute multi-hop CN
        self.cn_computer.precompute(graph, device)

        # ===== Main path =====
        if main_out is None:
            main_out = self.gcn_layer(graph, feat)  # Original behavior


        # ===== Compute gains for all hops =====
        hop_gains = []
        for i in range(self.num_hops):
            gain = self.cn_computer.aggregate(
                feat, hop_idx=i, transform_fn=self.hop_transforms[i]
            )
            hop_gains.append(gain)


        # ===== Key: concatenate router input first, then stack =====
        router_input = torch.cat([main_out] + hop_gains, dim=-1)  # [N, out_feats * (num_hops+1)]
        router_logits = self.router(router_input)  # [N, num_hops]

        # ===== Temperature scaling (consistent with minibatch) =====
        temp = F.softplus(self.router_temperature) + self.min_temperature
        router_logits = router_logits / temp
        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        # Now stack for subsequent weighting
        hop_gains = torch.stack(hop_gains, dim=1)  # [N, num_hops, out_feats]


        # Top-k selection
        if self.top_k < self.num_hops:
            top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_logits, dim=-1)

            sparse_weights = torch.zeros_like(router_logits)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            router_probs = sparse_weights
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        if self.training and self.router_dropout_p > 0:
            router_probs = F.dropout(
                router_probs, p=self.router_dropout_p, training=True
            )

        if self.fixed_hop is not None:
            moe_out = hop_gains[:, self.fixed_hop, :]
        else:
            moe_out = (hop_gains * router_probs.unsqueeze(-1)).sum(dim=1)  # [N, out_feats]

        # ===== Gated fusion =====
        output, gate_weights = self.fusion(main_out, moe_out)

        # Residual + LayerNorm
        output = self.layer_norm(output)


        # ===== Auxiliary loss =====
        if self.training:
            self._compute_aux_loss(router_logits, router_probs)

        # Statistics
        with torch.no_grad():
            # Record gate weights
            self._last_gate_weights = gate_weights.mean(dim=0).cpu().numpy()
            self._last_expert_stats = {
                'gcn_weight': gate_weights[:, 0].mean().item(),
                'moe_weight': gate_weights[:, 1].mean().item(),
            }
            for i in range(self.num_hops):
                self._last_expert_stats[f'hop{i+1}_avg_weight'] = router_probs[:, i].mean().item()

        return output

    def _compute_aux_loss(self, router_logits, router_probs):
        """Load balancing loss"""
        expert_usage = router_probs.mean(dim=0)
        router_prob_avg = F.softmax(router_logits, dim=-1).mean(dim=0)
        self.aux_loss = self.load_balance_weight * self.num_hops * (expert_usage * router_prob_avg).sum()


    def get_aux_loss(self):
        return self.aux_loss

    def get_expert_stats(self):
        return self._last_expert_stats

    def precompute_all(self, graph, feat):
        """
        Precompute CN aggregation results for all nodes on full graph (without applying transform)

        Args:
            graph: Full graph
            feat: [N, in_feats] Original features of full graph
        """
        device = feat.device

        # Reuse cn_computer.precompute
        self.cn_computer.precompute(graph, device)

        # Precompute aggregation results for each hop (without transform to avoid computation graph issues)
        self._precomputed_hop_feats = []
        with torch.no_grad():  # Key: don't create computation graph
            for i in range(self.num_hops):
                # Only do CN aggregation, don't apply hop_transforms
                hop_feat = self.cn_computer.aggregate(feat, hop_idx=i, transform_fn=None)
                self._precomputed_hop_feats.append(hop_feat)

        print(f"[MoE] Precomputation complete: {self.num_hops} hops")



    def forward_minibatch(self, feat, node_indices, main_out=None):
        """
        Mini-batch forward: index precomputed CN features

        Args:
            feat: [batch_size, in_feats] Original features of this batch
            node_indices: [batch_size] Node indices in full graph
            main_out: [batch_size, out_feats] Externally provided main path output (e.g., DGA convolution)

        Returns:
            [batch_size, out_feats]
        """
        # ===== Main path =====
        if main_out is None:
            # When no external input, use projection (maintain original behavior)
            main_out = self.input_proj(feat)

        # ===== Index precomputed hop features and apply transform =====
        hop_feats_raw = [
            self._precomputed_hop_feats[i][node_indices]
            for i in range(self.num_hops)
        ]
        hop_gains = [
            self.hop_transforms[i](hop_feats_raw[i])
            for i in range(self.num_hops)
        ]

        # ===== Routing (consistent with forward) =====
        router_input = torch.cat([main_out] + hop_gains, dim=-1)
        router_logits = self.router(router_input)
        # ===== Temperature scaling (higher temperature → more uniform) =====
        temp = F.softplus(self.router_temperature) + self.min_temperature
        router_logits = router_logits / temp
        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        # Top-k selection
        if self.top_k < self.num_hops:
            top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_logits, dim=-1)
            sparse_weights = torch.zeros_like(router_logits)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            router_probs = sparse_weights
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        if self.training and self.router_dropout_p > 0:
            router_probs = F.dropout(router_probs, p=self.router_dropout_p, training=True)

        # ===== Weighted aggregation (consistent with forward) =====
        hop_gains_stack = torch.stack(hop_gains, dim=1)  # [B, num_hops, out_feats]
        if self.fixed_hop is not None:
            moe_out = hop_gains_stack[:, self.fixed_hop, :]
        else:
            moe_out = (hop_gains_stack * router_probs.unsqueeze(-1)).sum(dim=1)  # [B, out_feats]

        output, gate_weights = self.fusion(main_out, moe_out)  # Two-way fusion
        output = self.layer_norm(output)

        # ===== Auxiliary loss =====
        if self.training:
            self._compute_aux_loss(router_logits, router_probs)

        # ===== Statistics (consistent with forward) =====
        with torch.no_grad():
            self._last_gate_weights = gate_weights.mean(dim=0).cpu().numpy()
            self._last_expert_stats = {
                'gcn_weight': gate_weights[:, 0].mean().item(),
                'moe_weight': gate_weights[:, 1].mean().item(),
            }
            for i in range(self.num_hops):
                self._last_expert_stats[f'hop{i+1}_avg_weight'] = router_probs[:, i].mean().item()

        return output
