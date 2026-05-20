"""
GNN_pd.py — Improved PDHeteroGNN with per-node-type input projections.

Change vs GNN.py:
    Added input projection layers — one linear layer per node type that
    maps each type's raw embedding (which may differ in dimension and
    semantic meaning) to a shared hidden space before message passing.

    Why this matters:
        - Compound nodes now use ChEMBL MT embeddings (chembl_dim)
        - All other nodes use TransE (400d)
        - Without projection, SAGEConv sees mixed-dimensional inputs
          and must implicitly learn to align them during message passing
        - With explicit per-type projections, each node type gets its
          own transformation to the shared hidden space first
        - This gives the model more capacity to learn type-specific
          transformations before aggregation begins

    The original GNN.py is unchanged — use it for the baseline comparison.
    This file is used by train_gnn_pd.py only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero


# ---------------------------------------------------------------------------
# Base GraphSAGE — identical to GNN.py
# ---------------------------------------------------------------------------

class BaseGraphSAGE(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 1
        self.drop  = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            out = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv((-1, -1), out))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Improved PDHeteroGNN with input projections
# ---------------------------------------------------------------------------

class PDHeteroGNN(nn.Module):
    """
    Heterogeneous GNN with per-node-type input projections.

    Parameters
    ----------
    metadata          : graph metadata from data.metadata()
    hidden_channels   : hidden dimension for message passing
    out_channels      : output embedding dimension (used in dot product)
    num_layers        : number of SAGEConv layers
    dropout           : dropout rate
    node_feature_dims : dict {ntype: raw_feature_dim}
                        if provided, adds a Linear(raw_dim → hidden_channels)
                        per node type before message passing
    """
    def __init__(self, metadata, hidden_channels: int = 128,
                 out_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.2,
                 node_feature_dims: dict | None = None):
        super().__init__()

        # NEW: per-node-type input projections
        # Maps each node type's raw embedding to hidden_channels
        # so that all node types enter message passing in the same space
        self.input_proj = nn.ModuleDict()
        if node_feature_dims:
            for ntype, in_dim in node_feature_dims.items():
                # Only add projection if raw dim != hidden_channels
                # (if they're already the same size, projection is redundant)
                if in_dim != hidden_channels:
                    self.input_proj[ntype] = nn.Linear(in_dim, hidden_channels)

        # Message passing encoder — same as before
        base = BaseGraphSAGE(hidden_channels, out_channels,
                             num_layers, dropout)
        self.encoder = to_hetero(base, metadata, aggr="sum")

    def encode(self, x_dict, edge_index_dict):
        """
        Step 1: project each node type to hidden_channels (if projection exists)
        Step 2: run heterogeneous message passing
        """
        projected = {}
        for ntype, x in x_dict.items():
            if ntype in self.input_proj:
                # Project + ReLU activation
                projected[ntype] = F.relu(self.input_proj[ntype](x))
            else:
                # No projection needed (already correct dimension)
                projected[ntype] = x
        return self.encoder(projected, edge_index_dict)

    def score_pairs(self, z_dict, edge_label_index,
                    src_type: str = "Compound",
                    dst_type: str = "Gene"):
        """
        Dot product scoring — measures embedding alignment.
        High score = model thinks this is a real interaction.
        """
        src_z = z_dict[src_type][edge_label_index[0]]
        dst_z = z_dict[dst_type][edge_label_index[1]]
        return (src_z * dst_z).sum(dim=-1)