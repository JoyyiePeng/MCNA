"""
ARC: Anomaly Detection with Residual Connections

This module implements ARC (Anomaly detection with Residual Connections),
a graph neural network that uses multi-hop features and cross-attention
mechanisms for anomaly detection tasks.

Key Features:
- Multi-hop feature extraction via graph propagation
- Residual connections between different hop features
- Cross-attention mechanism for few-shot learning
- Optional multi-hop CN enhancement via SparseMultiHopMoE
"""
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import scipy.sparse as sp
from models.ncn_modules import SparseMultiHopMoE


# ============================================================
# ARC Model
# ============================================================

class ARC(nn.Module):
    """
    Anomaly Detection with Residual Connections

    Architecture:
    1. Multi-layer MLP for feature encoding
    2. Optional multi-hop CN enhancement
    3. Residual feature computation across different hops
    4. Cross-attention for anomaly scoring

    Args:
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension (default: 1024)
        num_layers: Number of MLP layers (default: 4)
        drop_rate: Dropout rate (default: 0)
        activation: Activation function name (default: 'ELU')
        num_hops: Number of hops for multi-hop features (default: 2)
        use_multihop_moe: Whether to use multi-hop MoE enhancement (default: False)
        moe_num_hops: Number of hops for MoE (default: 3)
        moe_top_k: Top-k selection for MoE routing (default: 1)
    """
    def __init__(self, in_feats, h_feats=1024, num_layers=4, drop_rate=0, activation='ELU',
                 num_hops=2, use_multihop_moe=False, moe_num_hops=3, moe_top_k=1, **kwargs):
        super(ARC, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = num_hops
        self.h_feats = h_feats

        # Multi-hop MoE configuration
        self.use_multihop_moe = use_multihop_moe
        self.moe_num_hops = moe_num_hops
        self.moe_top_k = moe_top_k

        # Build MLP layers
        if num_layers == 0:
            return
        self.layers.append(nn.Linear(in_feats, h_feats))
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Linear(h_feats, h_feats))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.cross_attn = CrossAttn(h_feats * (num_hops - 1))

        # Initialize multi-hop MoE layer (optional)
        if use_multihop_moe:
            self.multihop_layer = SparseMultiHopMoE(
                gcn_layer=None,  # No built-in GCN
                in_feats=h_feats,  # MLP output dimension
                out_feats=h_feats,
                num_hops=moe_num_hops,
                top_k=moe_top_k,
                dropout=drop_rate,
                noise_std=0.2,
                load_balance_weight=0.01,
                cn_threshold=1,
                max_nodes_dense=10000,
                cache_dir=".cn_cache"
            )
            self._moe_precomputed = False

    def _precompute_moe(self, graph, feat, device):
        """
        Precompute multi-hop CN aggregation features

        Args:
            graph: DGL graph
            feat: MLP-encoded features [N, h_feats]
            device: Target device
        """
        if not self.use_multihop_moe or self._moe_precomputed:
            return

        self.multihop_layer.precompute_all(graph, feat.to(device))
        self._moe_precomputed = True

    def forward(self, h, graph=None):
        """
        Forward pass of ARC model

        Args:
            h: GraphData object with x_list attribute (multi-hop features)
            graph: DGL graph for multi-hop MoE (optional, required if use_multihop_moe=True)

        Returns:
            residual_embed: Residual embeddings [N, h_feats * (num_hops - 1)]
        """
        x_list = h.x_list

        # Z^{[l]} = MLP(X^{[l]})
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
            if i != len(self.layers) - 1:
                x_list = [self.act(x) for x in x_list]

        # Multi-hop MoE fusion (after MLP)
        if self.use_multihop_moe and graph is not None:
            device = x_list[0].device

            # Precompute (using first hop features, i.e., original node features through MLP)
            self._precompute_moe(graph, x_list[0], device)

            # Apply multi-hop MoE fusion to each hop's features
            x_list_fused = []
            for idx, x in enumerate(x_list):
                # Use full-graph forward (since ARC is full-batch)
                x_fused = self.multihop_layer.forward(graph, x, main_out=x)
                x_list_fused.append(x_fused)

            x_list = x_list_fused

        # Compute residual features
        residual_list = []
        # Z^{[0]}
        first_element = x_list[0]
        for h_i in x_list[1:]:
            # R^{[l]} = Z^{[l]} - Z^{[0]}
            dif = h_i - first_element
            residual_list.append(dif)

        # H = [R^{[1]} || ... || R^{[L]}]
        residual_embed = torch.hstack(residual_list)
        return residual_embed


# ============================================================
# Cross-Attention Module
# ============================================================

class CrossAttn(nn.Module):
    """
    Cross-attention mechanism for few-shot anomaly detection

    Uses query-key attention to compare query nodes with support nodes,
    enabling few-shot learning for anomaly detection.

    Args:
        embedding_dim: Dimension of node embeddings
    """
    def __init__(self, embedding_dim):
        super(CrossAttn, self).__init__()
        self.embedding_dim = embedding_dim

        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)

    def cross_attention(self, query_X, support_X):
        """
        Compute cross-attention between query and support nodes

        Args:
            query_X: Query node embeddings [N_query, embedding_dim]
            support_X: Support node embeddings [N_support, embedding_dim]

        Returns:
            weighted_query_embeddings: Attention-weighted query embeddings [N_query, embedding_dim]
        """
        Q = self.Wq(query_X)  # Query
        K = self.Wk(support_X)  # Key
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(
            torch.tensor(self.embedding_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_query_embeddings = torch.matmul(attention_weights, support_X)
        return weighted_query_embeddings

    def get_train_loss(self, X, y, num_prompt):
        """
        Compute training loss using cross-attention

        Args:
            X: Node embeddings [N, embedding_dim]
            y: Node labels [N]
            num_prompt: Number of support nodes to use

        Returns:
            loss: Cosine embedding loss
        """
        positive_indices = torch.nonzero((y == 1)).squeeze(1).tolist()
        all_negative_indices = torch.nonzero((y == 0)).squeeze(1).tolist()

        negative_indices = random.sample(all_negative_indices, len(positive_indices))
        # H_q_i, y_i == 1
        query_p_embed = X[positive_indices]
        # H_q_i, y_i == 0
        query_n_embed = X[negative_indices]
        # H_q
        query_embed = torch.vstack([query_p_embed, query_n_embed])

        remaining_negative_indices = list(set(all_negative_indices) - set(negative_indices))

        if len(remaining_negative_indices) < num_prompt:
            raise ValueError(f"Not enough remaining negative indices to select {num_prompt} support nodes.")

        support_indices = random.sample(remaining_negative_indices, num_prompt)
        support_indices = torch.tensor(support_indices).to(y.device)
        # H_k
        support_embed = X[support_indices]

        # The updated query node embeddings: \tilde{H_q}
        query_tilde_embeds = self.cross_attention(query_embed, support_embed)
        # tilde_p_embeds: \tilde{H_q_i}, y_i == 1; tilde_n_embeds: \tilde{H_q_i}, y_i == 0
        tilde_p_embeds, tilde_n_embeds = query_tilde_embeds[:len(positive_indices)], query_tilde_embeds[
                                                                                              len(positive_indices):]

        yp = torch.ones([len(negative_indices)]).to(y.device)
        yn = -torch.ones([len(positive_indices)]).to(y.device)
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, 1), if y_i == 0
        loss_qn = F.cosine_embedding_loss(query_n_embed, tilde_n_embeds, yp)
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, -1), if y_i == 1
        loss_qp = F.cosine_embedding_loss(query_p_embed, tilde_p_embeds, yn)
        loss = torch.mean(loss_qp + loss_qn)
        return loss

    def get_test_score(self, X, prompt_mask, y):
        """
        Compute anomaly scores for test nodes

        Args:
            X: Node embeddings [N, embedding_dim]
            prompt_mask: Boolean mask indicating support nodes [N]
            y: Node labels [N]

        Returns:
            query_score: Anomaly scores for query nodes [N_query]
        """
        # Prompt node indices
        negative_indices = torch.nonzero((prompt_mask == True) & (y == 0)).squeeze(1).tolist()
        n_support_embed = X[negative_indices]
        # Query node indices
        query_indices = torch.nonzero(prompt_mask == False).squeeze(1).tolist()
        # H_q
        query_embed = X[query_indices]
        # \tilde{H_q}
        query_tilde_embed = self.cross_attention(query_embed, n_support_embed)
        # dis(H_q, \tilde{H_q})
        diff = query_embed - query_tilde_embed
        # Score
        query_score = torch.sqrt(torch.sum(diff ** 2, dim=1))

        return query_score


# ============================================================
# Utility Functions
# ============================================================

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix

    Args:
        adj: Adjacency matrix (scipy sparse matrix)

    Returns:
        Normalized adjacency matrix (scipy sparse matrix)
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor

    Args:
        sparse_mx: Scipy sparse matrix

    Returns:
        PyTorch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
