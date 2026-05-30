"""
Heterogeneous GNN architecture for predicting interactions in a bipartite graph
of drugs (compounds) and protein targets (genes). Uses GraphSAGE message passing
with per-node-type input projections to handle embeddings of different dimensions.

Architecture:
  - BaseGraphSAGE: Standard multi-layer GraphSAGE encoder with dropout
  - PDHeteroGNN: Heterogeneous wrapper that projects each node type to a common
    hidden dimension before message passing, then scores drug-target pairs

Key components:
  1. Input projections (Linear layers): Map raw embedding dimensions to a uniform
     hidden_channels space. Only applied if raw dimension differs from target.
     This allows mixing embeddings from different sources (ChemBERTa drugs,
     ESM2 proteins) without dimension mismatch.

  2. Message passing (BaseGraphSAGE): Multi-layer SAGEConv with ReLU activations
     and dropout. Aggregates neighbor information using sum aggregation for
     heterogeneous graphs.

  3. Scoring (dot product): Measures alignment between drug and target embeddings.
     High scores indicate predicted interactions; used for training via BCE loss.

Node types:
  - "Compound": Drug nodes with ChemBERTa embeddings (512-dim)
  - "Gene": Protein target nodes with ESM2 embeddings (480-dim)

Parameters:
  hidden_channels: Dimension for message passing (default: 128)
  out_channels: Output embedding dimension for scoring (default: 64)
  num_layers: Number of GraphSAGE layers (default: 2)
  dropout: Dropout rate for regularization (default: 0.2)
  node_feature_dims: Dict mapping node types to raw embedding dimensions;
                     if provided, adds input projections for dimension mismatch

Dependencies:
  - torch, torch.nn: Neural network modules
  - torch_geometric: Graph neural network layers (SAGEConv, to_hetero)

Workflow:
  1. encode() projects inputs to hidden space, then applies GNN message passing
  2. score_pairs() computes dot product between drug and target node embeddings
  3. Training: BCE loss between scores and labels; inference: threshold scores

Usage example:
  data = HeteroData(...)  # Heterogeneous graph with node/edge types
  gnn = PDHeteroGNN(
      metadata=data.metadata(),
      node_feature_dims={"Compound": 512, "Gene": 480}
  )
  embeddings = gnn.encode(data.x_dict, data.edge_index_dict)
  scores = gnn.score_pairs(embeddings, edge_label_index)
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