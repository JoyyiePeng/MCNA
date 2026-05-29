"""
DGA-GNN: Dynamic Grouping Attention Graph Neural Network

This module implements DGA-GNN, a graph neural network with dynamic grouping
and attention mechanisms for graph anomaly detection tasks.

Key Features:
- Dynamic node grouping based on prediction confidence
- Multi-head attention for aggregating group-specific representations
- Intra-group convolution with edge weighting
- Optional multi-hop CN enhancement via SparseMultiHopMoE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as fn
from dgl.utils import expand_as_pair, dgl_warning
from models.ncn_modules import SparseMultiHopMoE


# ============================================================
# Graph Convolution Layer
# ============================================================

class IntraConv_single(nn.Module):
    """
    Intra-group graph convolution layer with edge weighting support

    Performs message passing within a group of nodes, with optional
    edge weights for weighted aggregation.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 add_self=True,
                 bias=True,
                 norm=None,
                 activation=None):
        super(IntraConv_single, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.add_self = add_self
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # Aggregator type: mean
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address backward compatibility issue brought by DGL #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from an old version of DGL, "
                        "DGL automatically converts it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def forward(self, graph, feat, etype=None, edge_weight=None):
        """
        Forward pass with optional edge weighting

        Args:
            graph: DGL graph or block
            feat: Node features [N, in_feats] or tuple of (src_feat, dst_feat)
            etype: Edge type (for heterogeneous graphs)
            edge_weight: Optional edge weights [E, 1]

        Returns:
            Updated node features [N, out_feats]
        """
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            msg_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.srcdata['degree'] = torch.ones((graph.num_src_nodes(), 1)).to(feat.device)
                graph.edata['_edge_weight'] = edge_weight
                msg_fn1 = fn.u_mul_e('h', '_edge_weight', 'm')
                msg_fn2 = fn.u_mul_e('degree', '_edge_weight', 'degree')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
            if edge_weight is not None:
                graph.update_all(msg_fn1, fn.sum('m', 'neigh'))
                graph.update_all(msg_fn2, fn.sum('degree', 'degree'))
                h_neigh = graph.dstdata['neigh'] / (graph.dstdata['degree'] + torch.FloatTensor([1e-8]).to(feat.device))
            else:
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']

            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)
            h_self = self.fc_self(h_self)

            if self.add_self:
                rst = h_self + h_neigh
            else:
                rst = h_neigh

            # Bias term
            if self.bias is not None:
                rst = rst + self.bias

            # Activation
            if self.activation is not None:
                rst = self.activation(rst)

            # Normalization
            if self.norm is not None:
                rst = self.norm(rst)

            return rst


# ============================================================
# DGA-GNN Model
# ============================================================

class DGA(nn.Module):
    """
    Dynamic Grouping Attention Graph Neural Network

    Architecture:
    1. Feature encoding with MLP
    2. Optional multi-hop CN enhancement
    3. Dynamic node grouping based on prediction confidence
    4. Group-specific graph convolutions
    5. Multi-head attention for aggregating group representations
    6. Final classification layer

    Args:
        in_feats: Input feature dimension
        n_hidden: Hidden dimension
        num_nodes: Total number of nodes in graph
        n_classes: Number of output classes
        n_etypes: Number of edge types
        p: Dropout probability
        n_head: Number of attention heads
        unclear_up: Upper threshold for unclear nodes
        unclear_down: Lower threshold for unclear nodes
        use_multihop: Whether to use multi-hop CN enhancement
        num_hops: Number of hops for multi-hop CN (default: 3)
        top_k: Top-k selection for multi-hop routing (default: 1)
    """
    def __init__(self, in_feats, n_hidden, num_nodes, n_classes, n_etypes, p=0.3, n_head=1,
                 unclear_up=0.1, unclear_down=0.1, use_multihop=False, num_hops=3, top_k=1):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_etypes = n_etypes
        self.unclear_up = unclear_up
        self.unclear_down = unclear_down
        self.register_buffer('super_mask', torch.ones((num_nodes, self.n_classes)))
        self.n_head = n_head
        self.use_multihop = use_multihop

        # Feature encoder
        hidden_units = [n_hidden, n_hidden]
        input_size = in_feats
        all_layers = []
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(nn.Dropout(p))
            all_layers.append(layer)
            all_layers.append(nn.BatchNorm1d(hidden_unit))
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
            self.last_dim = hidden_unit
        self.emb_layer = nn.Sequential(*all_layers)

        # Grouping classifier
        all_layers = []
        all_layers.append(nn.Linear(n_hidden, self.n_classes))
        self.emb_layer_fc = nn.Sequential(*all_layers)

        # Output layer
        all_layers = []
        all_layers.append(nn.Linear(self.n_head * n_hidden, n_hidden // 2))
        all_layers.append(nn.ReLU())
        all_layers.append(nn.Linear(n_hidden // 2, self.n_classes))
        self.final_fc_layer = nn.Sequential(*all_layers)

        # Attention mechanism
        self.attn_fn = nn.Tanh()
        self.W_f = nn.Sequential(nn.Linear(n_hidden, n_hidden * self.n_head), self.attn_fn)
        self.W_x = nn.Sequential(nn.Linear(n_hidden, n_hidden * self.n_head), self.attn_fn)

        self.reset_parameters()

        # Graph convolution layers
        if n_etypes == 1:
            intra_conv = IntraConv_single
        else:
            intra_conv = IntraConv_single  # Simplified for GADBench

        dgas = []
        for r in range(max(self.n_etypes, 1)):  # At least one edge type
            m = nn.ModuleDict({
                'all': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden),
                                  activation=nn.ReLU(), bias=False),
                'gp0': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden),
                                  activation=nn.ReLU(), bias=False, add_self=False),
                'gp1': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden),
                                  activation=nn.ReLU(), bias=False, add_self=False)
            })
            dgas.append(m)
        self.dgas = nn.ModuleList(dgas)

        # Multi-hop MoE layer (optional)
        if use_multihop:
            self.multihop_layer = SparseMultiHopMoE(
                gcn_layer=None,  # No built-in GCN, use external main_out
                in_feats=n_hidden,  # emb_layer output dimension
                out_feats=n_hidden,
                num_hops=num_hops,
                top_k=top_k,
                dropout=p,
                noise_std=0.2,
                load_balance_weight=0.01,
                cn_threshold=1,
                max_nodes_dense=10000,
                cache_dir=".cn_cache"
            )
        else:
            self.multihop_layer = None

    def reset_parameters(self):
        """Initialize model parameters"""
        gain = nn.init.calculate_gain('relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0)

    def precompute_multihop(self, graph, feat, device):
        """
        Precompute multi-hop CN aggregation features

        Args:
            graph: Full graph (CPU)
            feat: Original node features [N, in_feats]
            device: Target device
        """
        if not self.use_multihop:
            return

        # Move features to device and encode through emb_layer
        self.emb_layer.eval()  # Ensure BN uses running stats
        with torch.no_grad():
            # Process in batches to avoid OOM
            batch_size = 10000
            num_nodes = feat.shape[0]
            encoded_feats = []

            for start in range(0, num_nodes, batch_size):
                end = min(start + batch_size, num_nodes)
                batch_feat = feat[start:end].to(device)
                batch_encoded = self.emb_layer(batch_feat)
                encoded_feats.append(batch_encoded.cpu())

            full_encoded = torch.cat(encoded_feats, dim=0)

        # Precompute multi-hop aggregation (based on encoded features)
        self.multihop_layer.precompute_all(graph, full_encoded.to(device))
        self.emb_layer.train()  # Restore training mode

    def dynamic_grouping(self, mask, block, unclear_down, unclear_up):
        """
        Dynamically group nodes based on prediction confidence

        Args:
            mask: Node prediction probabilities [N, n_classes]
            block: DGL block
            unclear_down: Lower threshold for group 0 (confident negative)
            unclear_up: Upper threshold for group 1 (confident positive)

        Returns:
            mask0: Edge mask for group 0 (confident negative nodes)
            mask1: Edge mask for group 1 (confident positive nodes)
        """
        if block.number_of_edges() == 0:
            return torch.zeros(0).to(mask.device), torch.zeros(0).to(mask.device)

        edges = block.edges()
        # Use source node IDs to index mask
        if dgl.NID in block.srcdata:
            src_node_ids = block.srcdata[dgl.NID][edges[0].long()]
        else:
            src_node_ids = edges[0].long()

        mask0 = (mask[:, 1] <= unclear_down)[src_node_ids.long()].float()
        mask1 = (mask[:, 1] > unclear_up)[src_node_ids.long()].float()
        return mask0, mask1

    def forward(self, blocks, x):
        """
        Forward pass of DGA-GNN

        Args:
            blocks: List of DGL blocks for mini-batch training
            x: Input node features [N, in_feats]

        Returns:
            o: Final output logits [batch_size, n_classes]
            emb_out: Embedding layer output for auxiliary loss [batch_size, n_classes]
        """
        batch_size = blocks[-1].number_of_dst_nodes()

        # Feature encoding
        x = self.emb_layer(x)

        # Multi-hop fusion (after emb_layer)
        if self.use_multihop:
            # Get dst node indices in original graph
            node_indices = blocks[-1].dstdata[dgl.NID]

            # Apply multi-hop fusion to dst nodes
            # main_out: use encoded dst node features
            # feat: current batch dst node encoded features
            x_dst_fused = self.multihop_layer.forward_minibatch(
                feat=x[:batch_size],
                node_indices=node_indices,
                main_out=x[:batch_size],
            )

            # Replace dst features with fused features
            x = torch.cat([x_dst_fused, x[batch_size:]], dim=0)

        # Grouping classifier output
        emb_out = self.emb_layer_fc(x)

        mask0_dict = {}
        mask1_dict = {}
        block = blocks[0]

        # Dynamic grouping based on edge types
        if len(block.etypes) > 0:
            # Heterogeneous graph: iterate over all edge types
            for etype in block.etypes:
                edge_subgraph = block.edge_type_subgraph(etypes=[etype])
                mask0_dict[etype], mask1_dict[etype] = \
                    self.dynamic_grouping(self.super_mask, edge_subgraph,
                                          self.unclear_up, self.unclear_down)

            h_list = []
            for idx, etype in enumerate(block.etypes):
                h_list.append(self.dgas[idx]['all'](block, x, etype))
                h_list.append(self.dgas[idx]['gp0'](block, x, etype, mask0_dict[etype]))
                h_list.append(self.dgas[idx]['gp1'](block, x, etype, mask1_dict[etype]))
        else:
            # Homogeneous graph: use default processing
            mask0_dict['default'], mask1_dict['default'] = \
                self.dynamic_grouping(self.super_mask, block,
                                      self.unclear_up, self.unclear_down)

            h_list = []
            h_list.append(self.dgas[0]['all'](block, x, None))
            h_list.append(self.dgas[0]['gp0'](block, x, None, mask0_dict['default']))
            h_list.append(self.dgas[0]['gp1'](block, x, None, mask1_dict['default']))

        # Multi-head attention aggregation
        s_len = len(h_list)
        h_list = torch.stack(h_list, dim=1).contiguous()

        h_list_proj = self.W_f(h_list).view(batch_size, s_len, self.n_head, self.n_hidden)
        h_list_proj = h_list_proj.permute(0, 2, 1, 3).contiguous().view(-1, s_len, self.n_hidden)

        x_proj = self.W_x(x[:batch_size]).view(batch_size, self.n_head, self.n_hidden, 1)
        x_proj = x_proj.view(-1, self.n_hidden, 1)

        attention_logit = torch.bmm(h_list_proj, x_proj)
        soft_attention = F.softmax(attention_logit, dim=1).transpose(1, 2)
        h_list_rep = h_list.repeat([self.n_head, 1, 1])
        weighted_features = torch.bmm(soft_attention, h_list_rep).squeeze(-2)
        h = weighted_features.view(batch_size, -1)

        # Final classification
        o = self.final_fc_layer(h)

        return o, emb_out[:batch_size]

    def get_aux_loss(self):
        """Get auxiliary loss from multi-hop MoE layer"""
        if self.use_multihop and self.multihop_layer is not None:
            return self.multihop_layer.get_aux_loss()
        return 0.0
