import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from recsys_model import DualTowerRecModel
from recsys_data import DataSchema

# --- 1. Triplet Dataset ---
class TripletRecDataset(Dataset):
    def __init__(self, 
                 user_features, 
                 item_static_features, 
                 item_sequences, 
                 interactions,
                 all_item_ids):
        """
        Args:
            user_features: Dict of user tensors.
            item_static_features: Dict of item static tensors.
            item_sequences: Tensor of item sequences.
            interactions: DataFrame ['user_id', 'item_id'] (Positive interactions only).
            all_item_ids: List/Array of all available item IDs for negative sampling.
        """
        self.user_features = user_features
        self.item_static_features = item_static_features
        self.item_sequences = item_sequences
        self.interactions = interactions
        self.all_item_ids = all_item_ids
        
        # Group interactions by user for faster negative sampling check
        self.user_history = interactions.groupby('user_id')['item_id'].apply(set).to_dict()
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_idx = int(row['user_id'])
        pos_item_idx = int(row['item_id'])
        
        # Negative Sampling
        while True:
            neg_item_idx = np.random.choice(self.all_item_ids)
            if neg_item_idx not in self.user_history.get(user_idx, set()):
                break
                
        # Gather Features
        u_feats = {k: v[user_idx] for k, v in self.user_features.items()}
        
        pos_static = {k: v[pos_item_idx] for k, v in self.item_static_features.items()}
        pos_seq = self.item_sequences[pos_item_idx]
        
        neg_static = {k: v[neg_item_idx] for k, v in self.item_static_features.items()}
        neg_seq = self.item_sequences[neg_item_idx]
        
        return {
            'user': u_feats,
            'pos_item': {'static': pos_static, 'sequence': pos_seq},
            'neg_item': {'static': neg_static, 'sequence': neg_seq}
        }

# --- 2. BPR Loss ---
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pos_scores, neg_scores):
        # loss = -mean(log(sigmoid(pos - neg)))
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

# --- 3. Training Loop ---
def train_model(model, dataloader, epochs=10, lr=1e-3, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # L2 Reg
    criterion = BPRLoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move data to device if needed (skipping device logic for simplicity)
            user_batch = {'categorical': {}, 'numerical': None}
            # Reconstruct user batch structure expected by model
            # Assuming user features are split into categorical/numerical based on keys
            # For this demo, we'll hardcode the split based on the mock data keys
            user_batch['categorical'] = {k: v for k, v in batch['user'].items() if k in ['risk_preference', 'investment_goal']}
            # Numerical needs to be stacked: Age, Income
            # Note: In real app, DataSchema should handle this grouping better.
            # Here we assume 'age' and 'income' are the numericals
            if 'age' in batch['user'] and 'income' in batch['user']:
                user_batch['numerical'] = torch.cat([batch['user']['age'], batch['user']['income']], dim=1)
            
            pos_item_batch = batch['pos_item']
            neg_item_batch = batch['neg_item']
            
            # Forward
            pos_scores = model(user_batch, pos_item_batch)
            neg_scores = model(user_batch, neg_item_batch)
            
            # Loss
            loss = criterion(pos_scores, neg_scores)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Early Stopping (Mocking validation loss as training loss for simplicity)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# --- Verification ---
if __name__ == "__main__":
    # 1. Mock Data Setup (Same as recsys_data.py/recsys_model.py)
    # Users
    users_df = pd.DataFrame({
        'user_id': [0, 1, 2, 3],
        'age': [25, 40, 60, 30],
        'risk_preference': ['aggressive', 'moderate', 'conservative', 'moderate'],
        'investment_goal': ['growth', 'retirement', 'savings', 'growth'],
        'experience': ['novice', 'expert', 'average', 'novice'],
        'income': [50000, 120000, 80000, 60000]
    })
    u_feats = DataSchema.preprocess_user_features(users_df)
    
    # Items (10 items)
    num_items = 10
    prices = np.random.rand(num_items, 50, 1) * 100
    i_seq = DataSchema.preprocess_item_sequence(prices, window_size=30)
    i_static = {'category': torch.randint(0, 2, (num_items, 1))}
    all_item_ids = np.arange(num_items)
    
    # Interactions
    interactions = pd.DataFrame({
        'user_id': [0, 0, 1, 2, 3],
        'item_id': [0, 1, 2, 3, 4]
    })
    
    # Dataset & DataLoader
    dataset = TripletRecDataset(u_feats, i_static, i_seq, interactions, all_item_ids)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    
    # Model Setup
    user_config = {
        'embedding_dims': {'risk_preference': (3, 8), 'investment_goal': (3, 8)},
        'numerical_dim': 2, 
        'hidden_dims': [32],
        'output_dim': 16
    }
    item_config = {
        'embedding_dims': {'category': (2, 8)},
        'sequence_input_dim': 1,
        'hidden_dims': [32],
        'output_dim': 16
    }
    model = DualTowerRecModel(user_config, item_config)
    
    # Run Training
    print("Starting Training...")
    train_model(model, dataloader, epochs=5)
