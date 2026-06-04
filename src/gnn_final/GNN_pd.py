"""
Heterogeneous Graph Neural Network (PDHeteroGNN) Architecture.

How this script works mechanically:
1. GraphSAGE Base: Defines a multi-layer GraphSAGE encoder to perform message
   passing and neighborhood aggregation across a graph.
2. Heterogeneous Wrapper: Wraps the base encoder using PyTorch Geometric's
   `to_hetero` to handle multiple distinct node and edge types.
3. Dimensionality Projection: Includes an optional linear projection layer to
   map raw node features of varying dimensions into a unified hidden space
   before message passing occurs.
4. Link Prediction Scoring: Uses a dot-product scoring function to measure
   the alignment between source (Compound) and destination (Gene) node
   embeddings, outputting a continuous prediction score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

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


class PDHeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels: int = 128,
                 out_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.2,
                 node_feature_dims: dict | None = None):
        super().__init__()

        self.input_proj = nn.ModuleDict()
        if node_feature_dims:
            for ntype, in_dim in node_feature_dims.items():
                if in_dim != hidden_channels:
                    self.input_proj[ntype] = nn.Linear(in_dim, hidden_channels)

        base = BaseGraphSAGE(hidden_channels, out_channels,
                             num_layers, dropout)
        self.encoder = to_hetero(base, metadata, aggr="sum")

    def encode(self, x_dict, edge_index_dict):
        projected = {}
        for ntype, x in x_dict.items():
            if ntype in self.input_proj:
                projected[ntype] = F.relu(self.input_proj[ntype](x))
            else:
                projected[ntype] = x
        return self.encoder(projected, edge_index_dict)

    def score_pairs(self, z_dict, edge_label_index,
                    src_type: str = "Compound",
                    dst_type: str = "Gene"):
        src_z = z_dict[src_type][edge_label_index[0]]
        dst_z = z_dict[dst_type][edge_label_index[1]]
        return (src_z * dst_z).sum(dim=-1)