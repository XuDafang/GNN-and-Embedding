# update_embeddings.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from sklearn.preprocessing import LabelEncoder
import os

# --- Model and Helper Function Definitions from train_gnn.py ---
# These are needed to reconstruct the model before loading the saved weights.

def scatter_mean_manual(src, index, dim_size=None):
    """Manual implementation of scatter_mean using PyTorch operations"""
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    sum_result = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    
    sum_result.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones_like(src[:,:1]))
    
    count = torch.clamp(count, min=1)
    return sum_result / count

class WeightedSAGEConv(MessagePassing):
    """Custom GraphSAGE layer with average aggregation for weighted graphs"""
    def __init__(self, in_channels, out_channels, aggr='mean', **kwargs):
        super(WeightedSAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.weight_param = nn.Parameter(torch.Tensor(1, 1))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.constant_(self.weight_param, 1.0)
        if self.lin_self.bias is not None: nn.init.constant_(self.lin_self.bias, 0)
        if self.lin_neigh.bias is not None: nn.init.constant_(self.lin_neigh.bias, 0)
    
    def forward(self, x, edge_index, edge_weight=None):
        x_self = self.lin_self(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_neigh(out)
        return x_self + out
    
    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return x_j * edge_weight.view(-1, 1) * self.weight_param
        return x_j
    
    def aggregate(self, inputs, index, dim_size=None):
        return scatter_mean_manual(inputs, index, dim_size=dim_size)

class CustomGraphSAGE(nn.Module):
    """Custom GraphSAGE model supporting weighted graphs"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.1):
        super(CustomGraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(WeightedSAGEConv(in_dim, hidden_dim, aggr='mean'))
        for _ in range(num_layers - 2):
            self.convs.append(WeightedSAGEConv(hidden_dim, hidden_dim, aggr='mean'))
        self.convs.append(WeightedSAGEConv(hidden_dim, out_dim, aggr='mean'))
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.relu(x)
            x = self.dropout_layer(x)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

def fine_tune_model(model, data, epochs=30):
    """Training function to fine-tune the model."""
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5) # Use a smaller learning rate for fine-tuning
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index, data.edge_weight)
        
        pos_src, pos_dst = data.edge_index
        pos_scores = torch.sum(embeddings[pos_src] * embeddings[pos_dst], dim=1)
        pos_weights = data.edge_weight.view(-1) if data.edge_weight is not None else torch.ones_like(pos_scores)
        
        neg_dst = torch.randint(0, data.num_nodes, (pos_src.size(0),), device=data.x.device)
        neg_scores = torch.sum(embeddings[pos_src] * embeddings[neg_dst], dim=1)
        neg_weights = torch.ones_like(neg_scores) * 0.1
        
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        sample_weights = torch.cat([pos_weights, neg_weights])
        
        loss = F.binary_cross_entropy_with_logits(scores, labels, weight=sample_weights)
        loss.backward() # calculate the gradients for all parameters
        optimizer.step() # update the parameters based on the gradients
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

def update_embeddings():
    data_dir = 'data'
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load existing user data and the new engagement data
    print("Loading existing user map and new engagement data...")
    user_df = pd.read_csv(os.path.join(data_dir, 'users_with_indices.csv'))
    engagement_df = pd.read_csv(os.path.join(data_dir, 'engagements_1.csv'))

    num_nodes = len(user_df)
    id_to_idx = {row['id']: row['node_index'] for _, row in user_df.iterrows()}
    
    print(f"Users: {num_nodes}, New Engagements: {len(engagement_df)}")

    # 2. Calculate composite edge weight from new engagement data
    # The training script expects a single 'weight' column. We create it by summing engagement types.
    engagement_types = ['like', 'save', 'reshare', 'privateShare', 'videoView60', 'profileVisit', 'follow']
    engagement_df['weight'] = engagement_df[engagement_types].sum(axis=1)
    
    print("New edge weight statistics:")
    print(f"Min: {engagement_df['weight'].min():.3f}, Max: {engagement_df['weight'].max():.3f}, Mean: {engagement_df['weight'].mean():.3f}")

    # 3. Reconstruct the graph with updated edges
    print("Reconstructing graph with updated data...")
    valid_mask = (engagement_df['actor_id'].isin(id_to_idx)) & (engagement_df['receiver_id'].isin(id_to_idx))
    valid_engagements = engagement_df[valid_mask].copy()

    source_nodes = valid_engagements['actor_id'].map(id_to_idx).values
    target_nodes = valid_engagements['receiver_id'].map(id_to_idx).values
    
    edge_index = torch.tensor(np.stack([source_nodes, target_nodes]), dtype=torch.long)
    edge_weights = valid_engagements['weight'].values
    edge_weights_normalized = edge_weights / edge_weights.max()
    edge_weight = torch.tensor(edge_weights_normalized, dtype=torch.float).view(-1, 1)

    # Recreate node features exactly as in the original training
    tag_encoder = LabelEncoder()
    tag_encoded = tag_encoder.fit_transform(user_df['tag'])
    tag_onehot = np.eye(len(tag_encoder.classes_))[tag_encoded]
    
    follower_normalized = user_df[['followerCount']] / max(user_df['followerCount'])
    node_features = np.hstack([tag_onehot, follower_normalized.values])
    x = torch.tensor(node_features, dtype=torch.float)

    # Create the updated graph data object
    updated_graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    print(f"Updated graph: Nodes: {updated_graph_data.num_nodes}, Edges: {updated_graph_data.num_edges}")

    # 4. Load the pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    in_channels = updated_graph_data.num_node_features
    hidden_channels = 64
    out_channels = 32

    model = CustomGraphSAGE(in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1)
    
    model_path = os.path.join(data_dir, 'weighted_gnn_model.pth')
    print(f"Loading pre-trained model from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # 5. Fine-tune the model on the new data
    print("Starting fine-tuning with updated engagement weights...")
    updated_graph_data = updated_graph_data.to(device)
    model = fine_tune_model(model, updated_graph_data, epochs=30)

    # 6. Save the updated results
    print("Saving updated artifacts...")
    model.eval()
    with torch.no_grad():
        updated_embeddings = model(updated_graph_data.x, updated_graph_data.edge_index, updated_graph_data.edge_weight).cpu().numpy()

    torch.save(model.state_dict(), os.path.join(data_dir, 'updated_edges_model.pth'))
    np.save(os.path.join(data_dir, 'updated_edges_embeddings.npy'), updated_embeddings)

    print("Fine-tuning and embedding update completed successfully!")
    print(f"Updated embeddings shape: {updated_embeddings.shape}")
    print(f"Updated model saved: updated_gnn_model.pth")
    print(f"Updated embeddings saved: updated_final_embeddings.npy")

if __name__ == '__main__':
    update_embeddings()