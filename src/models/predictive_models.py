from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Part 3.1: Random Forest Classifier on Concatenated Embeddings
class ConcatenatedClassifier:
    def __init__(self, n_estimators=100):
        self.rf = RandomForestClassifier(n_estimators=n_estimators)
        
    def prepare_features(self, drug_vec, target_vec, phenotype_vec):
        # Concatenate drug, target, and phenotype vectors
        # Shape: (batch, dim_drug + dim_target + dim_pheno)
        return torch.cat([drug_vec, target_vec, phenotype_vec], dim=1).numpy()
    
    def train(self, features, labels):
        self.rf.fit(features, labels)
        
    def predict(self, features):
        return self.rf.predict(features)

# Part 3.2: Graph Neural Network (GNN) for Systems Biology
class ParkinsonGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ParkinsonGNN, self).__init__()
        # Using Graph Convolutional Networks (GCN) as a base
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        # 1. Update node representations via message passing
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # 2. Global pooling to capture systems-level topology
        x = global_mean_pool(x, batch)
        
        # 3. Final classification head
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

if __name__ == "__main__":
    # Dummy shapes
    batch_size = 5
    drug_dim, target_dim, pheno_dim = 256, 1280, 4096
    
    drug_vec = torch.randn(batch_size, drug_dim)
    target_vec = torch.randn(batch_size, target_dim)
    pheno_vec = torch.randn(batch_size, pheno_dim)
    
    # 3.1 Test RF Concatenation
    clf = ConcatenatedClassifier()
    features = clf.prepare_features(drug_vec, target_vec, pheno_vec)
    print(f"Concatenated feature shape: {features.shape}") # Should be (5, 256+1280+4096) = (5, 5632)
    
    # 3.2 Test GNN
    # GNN expects (num_nodes, in_channels)
    gnn = ParkinsonGNN(in_channels=128, hidden_channels=64, out_channels=1)
    print(f"GNN initialized.")
