"""
GNN.py  —  PDHeteroGNN with dot product predictor.

Key change from previous version:
    LinkPredictor (MLP) replaced with dot product scoring.
    GNNDRKG uses HeteroDotProductPredictor — dot product directly
    measures embedding alignment, which is standard for knowledge
    graph link prediction and outperforms MLP on this task.
"""

import torch
import torch.nn as nn

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
                 dropout: float = 0.2):
        super().__init__()
        base = BaseGraphSAGE(hidden_channels, out_channels, num_layers, dropout)
        self.encoder = to_hetero(base, metadata, aggr="sum")

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def score_pairs(self, z_dict, edge_label_index,
                    src_type: str = "Compound", dst_type: str = "Disease"):
        """
        Dot product scoring — same as GNNDRKG's HeteroDotProductPredictor.
        src and dst type default to Compound/Disease to match GNNDRKG's
        prediction edge type: ('Compound', 'DRUGBANK::treats::...', 'Disease')
        """
        src_z = z_dict[src_type][edge_label_index[0]]
        dst_z = z_dict[dst_type][edge_label_index[1]]
        return (src_z * dst_z).sum(dim=-1)  # dot product, shape (n_pairs,)

def _validate_edge_index(edge_index, src_nodes, dst_nodes, edge_name):
    if edge_index.numel() == 0:
        return
    if edge_index[0].max() >= src_nodes or edge_index[1].max() >= dst_nodes:
        raise ValueError(
            f"{edge_name} out of bounds: "
            f"src_max={edge_index[0].max()} (n={src_nodes}), "
            f"dst_max={edge_index[1].max()} (n={dst_nodes})"
        )


def build_pd_graph(
    drug_x: torch.Tensor,
    target_x: torch.Tensor,
    drug_target_edge_index: torch.Tensor,
    phenotype_x: torch.Tensor | None = None,
    target_phenotype_edge_index: torch.Tensor | None = None,
    add_reverse_edges: bool = True,
) -> HeteroData:
    data = HeteroData()
    data["drug"].x   = drug_x
    data["target"].x = target_x

    _validate_edge_index(drug_target_edge_index,
                         drug_x.size(0), target_x.size(0), "drug->target")
    data["drug", "binds_to", "target"].edge_index = drug_target_edge_index

    if add_reverse_edges:
        data["target", "rev_binds_to", "drug"].edge_index = torch.stack(
            [drug_target_edge_index[1], drug_target_edge_index[0]], dim=0)

    if phenotype_x is not None and target_phenotype_edge_index is not None:
        data["phenotype"].x = phenotype_x
        _validate_edge_index(target_phenotype_edge_index,
                             target_x.size(0), phenotype_x.size(0), "target->phenotype")
        data["target", "associated_with", "phenotype"].edge_index = (
            target_phenotype_edge_index)
        if add_reverse_edges:
            data["phenotype", "rev_associated_with", "target"].edge_index = torch.stack(
                [target_phenotype_edge_index[1], target_phenotype_edge_index[0]], dim=0)

    return data