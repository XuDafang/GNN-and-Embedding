import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def scatter_mean_manual(src, index, dim_size=None):
    """Manual implementation of scatter_mean using PyTorch operations"""
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    # Sum values by index
    sum_result = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    
    # Use index_add_ for efficient summation
    sum_result.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones_like(src[:,:1]))
    
    # Avoid division by zero
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
        if self.lin_self.bias is not None:
            nn.init.constant_(self.lin_self.bias, 0)
        if self.lin_neigh.bias is not None:
            nn.init.constant_(self.lin_neigh.bias, 0)
    
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
    """Custom GraphSAGE model supporting weighted graphs with average aggregation"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.1):
        super(CustomGraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
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

def create_edge_weights(engagement_df, weight_strategy='engagement_count'):
    """Create edge weights based on different strategies"""
    
    if weight_strategy == 'engagement_count':
        weight_df = engagement_df.groupby(['actor_id', 'receiver_id']).size().reset_index(name='weight')
        if weight_df['weight'].max() > 0:
            weight_df['weight'] = weight_df['weight'] / weight_df['weight'].max()
        return weight_df
    
    elif weight_strategy == 'binary':
        weight_df = engagement_df[['actor_id', 'receiver_id']].drop_duplicates()
        weight_df['weight'] = 1.0
        return weight_df
    
    elif weight_strategy == 'reciprocal':
        engagement_pairs = engagement_df[['actor_id', 'receiver_id']].drop_duplicates()
        reciprocal_weights = []
        
        for _, row in engagement_pairs.iterrows():
            actor, receiver = row['actor_id'], row['receiver_id']
            reverse_exists = ((engagement_df['actor_id'] == receiver) & 
                            (engagement_df['receiver_id'] == actor)).any()
            weight = 2.0 if reverse_exists else 1.0
            reciprocal_weights.append(weight)
        
        weight_df = engagement_pairs.copy()
        weight_df['weight'] = reciprocal_weights
        if weight_df['weight'].max() > 0:
            weight_df['weight'] = weight_df['weight'] / weight_df['weight'].max()
        return weight_df
    
    else:
        raise ValueError(f"Unknown weight strategy: {weight_strategy}")

def train_gnn():
    data_dir = 'data'
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading data...")
    user_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
    engagement_df = pd.read_csv(os.path.join(data_dir, 'engagements.csv'))

    print(f"Users: {len(user_df)}, Engagements: {len(engagement_df)}")

    # Create consistent node mapping
    unique_users = sorted(user_df['id'].unique())
    num_nodes = len(unique_users)
    id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    
    print(f"Unique users: {num_nodes}")

    # Filter engagements
    valid_mask = (engagement_df['actor_id'].isin(id_to_idx)) & (engagement_df['receiver_id'].isin(id_to_idx))
    valid_engagements = engagement_df[valid_mask].copy()
    print(f"Valid engagements: {len(valid_engagements)}")

    # Create edge index 
    source_nodes = valid_engagements['actor_id'].map(id_to_idx).values
    target_nodes = valid_engagements['receiver_id'].map(id_to_idx).values
    
    # Create single numpy array first, then convert to tensor
    edge_index_np = np.stack([source_nodes, target_nodes], axis=0)
    edge_index = torch.from_numpy(edge_index_np).long()

    print(f"Edge index shape: {edge_index.shape}")

    # Create edge weights
    print("Creating edge weights...")
    weight_df = create_edge_weights(valid_engagements, weight_strategy='engagement_count')
    
    # Map weights to edge indices
    edge_weight_map = {(row['actor_id'], row['receiver_id']): row['weight'] 
                      for _, row in weight_df.iterrows()}
    
    edge_weights = []
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        src_id = unique_users[src_idx]
        dst_id = unique_users[dst_idx]
        weight = edge_weight_map.get((src_id, dst_id), 0.0)
        edge_weights.append(weight)
    
    edge_weight = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    print(f"Edge weights - Min: {edge_weight.min():.3f}, Max: {edge_weight.max():.3f}, Mean: {edge_weight.mean():.3f}")

    # Create node features
    user_df_sorted = user_df.set_index('id').loc[unique_users].reset_index()
    
    # One-hot encode tags
    tag_encoder = LabelEncoder()
    tag_encoded = tag_encoder.fit_transform(user_df_sorted['tag'])
    tag_onehot = np.eye(len(tag_encoder.classes_))[tag_encoded]
    
    # Normalize follower count
    scaler = StandardScaler()
    follower_normalized = scaler.fit_transform(user_df_sorted[['followerCount']])
    
    # Combine features
    node_features = np.hstack([tag_onehot, follower_normalized])
    x = torch.from_numpy(node_features).float()
    
    print(f"Node features shape: {x.shape}")
    print(f"Number of tags: {len(tag_encoder.classes_)}")

    # Verify dimensions
    assert x.shape[0] == num_nodes, f"Node feature mismatch: {x.shape[0]} vs {num_nodes}"
    assert edge_index.max() < num_nodes, f"Edge index out of bounds: {edge_index.max()} >= {num_nodes}"

    # Create graph data with weights
    graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    
    print("Weighted graph data created successfully!")
    print(f"Nodes: {graph_data.num_nodes}, Features: {graph_data.num_node_features}, Edges: {graph_data.num_edges}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model parameters
    in_channels = graph_data.num_node_features
    hidden_channels = 64
    out_channels = 32
    
    print(f"Model dimensions: in={in_channels}, hidden={hidden_channels}, out={out_channels}")

    # Use custom GraphSAGE model
    model = CustomGraphSAGE(in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Move data to device
    graph_data = graph_data.to(device)

    # Training function with weighted loss
    def train_model(model, data, epochs=50):
        model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass with edge weights
            embeddings = model(data.x, data.edge_index, data.edge_weight)
            
            # Positive samples (existing edges)
            pos_src, pos_dst = data.edge_index
            pos_scores = torch.sum(embeddings[pos_src] * embeddings[pos_dst], dim=1)
            
            # Use edge weights for positive sample weighting
            pos_weights = data.edge_weight.view(-1) if data.edge_weight is not None else torch.ones_like(pos_scores)
            
            # Negative samples (random pairs)
            neg_dst = torch.randint(0, data.num_nodes, (pos_src.size(0),), device=device)
            neg_scores = torch.sum(embeddings[pos_src] * embeddings[neg_dst], dim=1)
            neg_weights = torch.ones_like(neg_scores) * 0.1  # Lower weight for negative samples
            
            # Combine and compute weighted loss
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            sample_weights = torch.cat([pos_weights, neg_weights])
            
            # Weighted BCE loss
            loss = F.binary_cross_entropy_with_logits(scores, labels, weight=sample_weights)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        
        return model

    print("Starting training with weighted GraphSAGE...")
    model = train_model(model, graph_data, epochs=50)

    # Save results
    model.eval()
    with torch.no_grad():
        embeddings = model(graph_data.x, graph_data.edge_index, graph_data.edge_weight).cpu().numpy()

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(data_dir, 'weighted_gnn_model.pth'))
    np.save(os.path.join(data_dir, 'weighted_final_embeddings.npy'), embeddings)
    
    # Save mapping and weights
    user_df_sorted['node_index'] = range(num_nodes)
    user_df_sorted.to_csv(os.path.join(data_dir, 'users_with_indices.csv'), index=False)
    
    # Save edge weights info
    weight_info = pd.DataFrame({
        'source': edge_index[0].cpu().numpy(),
        'target': edge_index[1].cpu().numpy(),
        'weight': edge_weight.cpu().numpy().flatten()
    })
    weight_info.to_csv(os.path.join(data_dir, 'edge_weights.csv'), index=False)

    print("Weighted GraphSAGE training completed successfully!")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Model saved: weighted_gnn_model.pth")
    print(f"Edge weights saved: edge_weights.csv")

if __name__ == '__main__':
    train_gnn()