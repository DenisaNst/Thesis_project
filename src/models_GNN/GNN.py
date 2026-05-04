import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, to_hetero
except ImportError as exc:
    raise ImportError("torch-geometric is required. Run: pip install torch-geometric") from exc


class BaseGraphSAGE(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug_z, target_z):
        pair = torch.cat([drug_z, target_z], dim=-1)
        return self.mlp(pair).squeeze(-1)


class PDHeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels: int = 256, out_channels: int = 128):
        super().__init__()
        base = BaseGraphSAGE(hidden_channels=hidden_channels, out_channels=out_channels)
        self.encoder = to_hetero(base, metadata, aggr="sum")
        self.predictor = LinkPredictor(emb_dim=out_channels)

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def score_pairs(self, z_dict, edge_label_index):
        # edge_label_index shape: [2, num_pairs]
        d_idx = edge_label_index[0]
        t_idx = edge_label_index[1]
        drug_z = z_dict["drug"][d_idx]
        target_z = z_dict["target"][t_idx]
        return self.predictor(drug_z, target_z)


def _validate_edge_index(edge_index, src_nodes, dst_nodes, edge_name):
    if edge_index.numel() == 0:
        return
    src_max = int(edge_index[0].max().item())
    dst_max = int(edge_index[1].max().item())
    if src_max >= src_nodes or dst_max >= dst_nodes:
        raise ValueError(
            f"{edge_name} edge_index out of bounds. "
            f"src_max={src_max} (n_src={src_nodes}), dst_max={dst_max} (n_dst={dst_nodes})"
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
    data["drug"].x = drug_x
    data["target"].x = target_x

    _validate_edge_index(
        drug_target_edge_index, drug_x.size(0), target_x.size(0), "drug->target"
    )
    data["drug", "binds_to", "target"].edge_index = drug_target_edge_index

    if add_reverse_edges:
        rev_dt = torch.stack([drug_target_edge_index[1], drug_target_edge_index[0]], dim=0)
        data["target", "rev_binds_to", "drug"].edge_index = rev_dt

    # Optional phenotype branch
    if phenotype_x is not None and target_phenotype_edge_index is not None:
        data["phenotype"].x = phenotype_x
        _validate_edge_index(
            target_phenotype_edge_index, target_x.size(0), phenotype_x.size(0), "target->phenotype"
        )
        data["target", "associated_with", "phenotype"].edge_index = target_phenotype_edge_index
        if add_reverse_edges:
            rev_tp = torch.stack(
                [target_phenotype_edge_index[1], target_phenotype_edge_index[0]], dim=0
            )
            data["phenotype", "rev_associated_with", "target"].edge_index = rev_tp

    return data
