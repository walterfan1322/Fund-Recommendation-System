import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# --- Data Schemas (Mock Data Generators for now) ---

class DataSchema:
    """Defines the structure and preprocessing for different data types."""
    
    @staticmethod
    def preprocess_user_features(user_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Preprocesses user features.
        Expected columns: ['age', 'risk_preference', 'investment_goal', 'experience', 'income']
        """
        # 1. Numerical: Age, Income (Normalize)
        age = torch.tensor(user_df['age'].values, dtype=torch.float32).unsqueeze(1)
        age = (age - age.mean()) / (age.std() + 1e-6)
        
        income = torch.tensor(user_df['income'].values, dtype=torch.float32).unsqueeze(1)
        income = torch.log1p(income) # Log transform for income
        
        # 2. Categorical: Risk, Goal, Experience (Map to Indices)
        # Mappings (Simplified for demo)
        risk_map = {'conservative': 0, 'moderate': 1, 'aggressive': 2}
        goal_map = {'savings': 0, 'growth': 1, 'retirement': 2}
        exp_map = {'novice': 0, 'average': 1, 'expert': 2}
        
        risk = torch.tensor([risk_map.get(x, 1) for x in user_df['risk_preference']], dtype=torch.long).unsqueeze(1)
        goal = torch.tensor([goal_map.get(x, 1) for x in user_df['investment_goal']], dtype=torch.long).unsqueeze(1)
        exp = torch.tensor([exp_map.get(x, 0) for x in user_df['experience']], dtype=torch.long).unsqueeze(1)
        
        return {
            'age': age,
            'income': income,
            'risk_preference': risk,
            'investment_goal': goal,
            'experience': exp
        }

    @staticmethod
    def preprocess_item_sequence(price_history: np.ndarray, window_size: int = 30) -> torch.Tensor:
        """
        Preprocesses time series data (Prices/NAV).
        Input: (N_Items, Sequence_Length, Features)
        Output: (N_Items, Window_Size, Features) - Normalized returns
        """
        # Simplified: Just take the last 'window_size' steps and calculate returns
        # In real scenario, we might slide the window.
        
        # Assuming price_history is (N, Time, Feats)
        # Take last window_size + 1 to calc returns
        if price_history.shape[1] <= window_size:
             # Pad if too short (Not implemented for brevity, assuming sufficient length)
             pass
             
        recent_prices = price_history[:, -(window_size+1):, :]
        
        # Calculate returns: (P_t - P_{t-1}) / P_{t-1}
        # Avoid division by zero
        returns = np.diff(recent_prices, axis=1) / (recent_prices[:, :-1, :] + 1e-6)
        
        return torch.tensor(returns, dtype=torch.float32)

# --- PyTorch Dataset ---

class RecSysDataset(Dataset):
    def __init__(self, 
                 user_features: Dict[str, torch.Tensor], 
                 item_static_features: Dict[str, torch.Tensor],
                 item_sequences: torch.Tensor,
                 interactions: pd.DataFrame):
        """
        Args:
            user_features: Preprocessed user tensors (by User ID index).
            item_static_features: Preprocessed item static tensors (by Item ID index).
            item_sequences: Preprocessed item time series tensors (by Item ID index).
            interactions: DataFrame with ['user_id', 'item_id', 'label'].
        """
        self.user_features = user_features
        self.item_static_features = item_static_features
        self.item_sequences = item_sequences
        self.interactions = interactions
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_idx = int(row['user_id'])
        item_idx = int(row['item_id'])
        label = float(row['label'])
        
        # Gather User Features
        u_feats = {k: v[user_idx] for k, v in self.user_features.items()}
        
        # Gather Item Features
        i_static = {k: v[item_idx] for k, v in self.item_static_features.items()}
        i_seq = self.item_sequences[item_idx]
        
        return {
            'user': u_feats,
            'item_static': i_static,
            'item_seq': i_seq,
            'target': torch.tensor(label, dtype=torch.float32)
        }

# --- Example Usage / Verification ---

if __name__ == "__main__":
    # 1. Mock Data
    users = pd.DataFrame({
        'user_id': [0, 1, 2],
        'age': [25, 40, 60],
        'risk_preference': ['aggressive', 'moderate', 'conservative'],
        'investment_goal': ['growth', 'retirement', 'savings'],
        'experience': ['novice', 'expert', 'average'],
        'income': [50000, 120000, 80000]
    })
    
    # Mock Item Data (2 items, 50 days, 1 feature (Close price))
    # Item 0: Rising, Item 1: Volatile
    prices = np.random.rand(2, 50, 1) * 100 
    
    # Mock Interactions
    interactions = pd.DataFrame({
        'user_id': [0, 0, 1, 2],
        'item_id': [0, 1, 0, 1],
        'label': [1, 0, 1, 1] # 1 = Click/Buy
    })
    
    # 2. Preprocessing
    u_feats = DataSchema.preprocess_user_features(users)
    
    # Item Static (Mock)
    i_static = {
        'category': torch.tensor([0, 1], dtype=torch.long) # 0: Stock, 1: Fund
    }
    
    # Item Sequence
    i_seq = DataSchema.preprocess_item_sequence(prices, window_size=30)
    
    # 3. Dataset & DataLoader
    dataset = RecSysDataset(u_feats, i_static, i_seq, interactions)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 4. Iterate
    print("Iterating DataLoader:")
    for batch in dataloader:
        print("User Age Batch:", batch['user']['age'])
        print("Item Seq Shape:", batch['item_seq'].shape)
        print("Target:", batch['target'])
        break
