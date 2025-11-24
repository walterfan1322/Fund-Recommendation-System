import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, embedding_dims, numerical_dim, hidden_dims, output_dim):
        """
        Args:
            embedding_dims: Dict {feature_name: (num_embeddings, embedding_dim)}
            numerical_dim: Number of numerical features
            hidden_dims: List of hidden layer sizes for MLP
            output_dim: Size of the final user vector
        """
        super().__init__()
        
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num, dim) 
            for name, (num, dim) in embedding_dims.items()
        })
        
        total_input_dim = sum(dim for _, dim in embedding_dims.values()) + numerical_dim
        
        layers = []
        in_dim = total_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(0.1))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, categorical_inputs, numerical_inputs):
        """
        Args:
            categorical_inputs: Dict {feature_name: Tensor(Batch, 1)}
            numerical_inputs: Tensor(Batch, Num_Features)
        """
        emb_list = []
        for name, emb_layer in self.embeddings.items():
            # Squeeze to (Batch,) for Embedding layer if input is (Batch, 1)
            inp = categorical_inputs[name].squeeze(1) 
            emb_list.append(emb_layer(inp))
            
        # Concat embeddings and numerical features
        x = torch.cat(emb_list + [numerical_inputs], dim=1)
        return self.mlp(x)

class ItemTower(nn.Module):
    def __init__(self, embedding_dims, sequence_input_dim, hidden_dims, output_dim):
        """
        Args:
            embedding_dims: Dict {feature_name: (num_embeddings, embedding_dim)}
            sequence_input_dim: Number of features in time series (e.g., Price, Volume)
            hidden_dims: List of hidden layer sizes for MLP
            output_dim: Size of the final item vector
        """
        super().__init__()
        
        # Static Features Embedding
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num, dim) 
            for name, (num, dim) in embedding_dims.items()
        })
        
        # Sequence Encoder (LSTM)
        self.lstm = nn.LSTM(input_size=sequence_input_dim, hidden_size=64, batch_first=True)
        
        # MLP
        total_static_dim = sum(dim for _, dim in embedding_dims.values())
        total_input_dim = total_static_dim + 64 # 64 is LSTM hidden size
        
        layers = []
        in_dim = total_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, static_inputs, sequence_inputs):
        """
        Args:
            static_inputs: Dict {feature_name: Tensor(Batch, 1)}
            sequence_inputs: Tensor(Batch, Seq_Len, Features)
        """
        # 1. Process Static
        emb_list = []
        for name, emb_layer in self.embeddings.items():
            inp = static_inputs[name].squeeze(1)
            emb_list.append(emb_layer(inp))
        static_vec = torch.cat(emb_list, dim=1)
        
        # 2. Process Sequence
        # LSTM output: (Batch, Seq, Hidden), (h_n, c_n)
        # We take the last hidden state: h_n[-1]
        _, (h_n, _) = self.lstm(sequence_inputs)
        seq_vec = h_n[-1] # (Batch, Hidden)
        
        # 3. Concat and MLP
        x = torch.cat([static_vec, seq_vec], dim=1)
        return self.mlp(x)

class DualTowerRecModel(nn.Module):
    def __init__(self, user_config, item_config):
        super().__init__()
        self.user_tower = UserTower(**user_config)
        self.item_tower = ItemTower(**item_config)
        
    def forward(self, user_batch, item_batch):
        """
        Args:
            user_batch: Dict containing 'categorical' and 'numerical'
            item_batch: Dict containing 'static' and 'sequence'
        """
        u_vec = self.user_tower(user_batch['categorical'], user_batch['numerical'])
        i_vec = self.item_tower(item_batch['static'], item_batch['sequence'])
        
        # Normalize vectors for Cosine Similarity (Optional but recommended)
        u_vec = F.normalize(u_vec, p=2, dim=1)
        i_vec = F.normalize(i_vec, p=2, dim=1)
        
        # Dot Product -> Score
        # (Batch, Dim) * (Batch, Dim) -> (Batch, 1)
        score = (u_vec * i_vec).sum(dim=1, keepdim=True)
        return score

# --- Verification ---
if __name__ == "__main__":
    # Mock Config
    user_config = {
        'embedding_dims': {'risk_preference': (3, 8), 'investment_goal': (3, 8)},
        'numerical_dim': 2, # Age, Income
        'hidden_dims': [32],
        'output_dim': 16
    }
    
    item_config = {
        'embedding_dims': {'category': (2, 8)}, # Stock/Fund
        'sequence_input_dim': 1, # Price only
        'hidden_dims': [32],
        'output_dim': 16
    }
    
    model = DualTowerRecModel(user_config, item_config)
    print(model)
    
    # Mock Input
    batch_size = 4
    user_batch = {
        'categorical': {
            'risk_preference': torch.randint(0, 3, (batch_size, 1)),
            'investment_goal': torch.randint(0, 3, (batch_size, 1))
        },
        'numerical': torch.randn(batch_size, 2)
    }
    
    item_batch = {
        'static': {
            'category': torch.randint(0, 2, (batch_size, 1))
        },
        'sequence': torch.randn(batch_size, 30, 1) # 30 days, 1 feature
    }
    
    # Forward Pass
    score = model(user_batch, item_batch)
    print("\nOutput Score Shape:", score.shape)
    print("Scores:", score.detach().numpy())
