import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader

try:
    import faiss
except ImportError:
    faiss = None

class ItemDataset(Dataset):
    def __init__(self, num_items=100):
        self.num_items = num_items
        self.static = {'category': torch.randint(0, 2, (num_items, 1))}
        self.sequence = torch.randn(num_items, 30, 1)
        self.ids = torch.arange(num_items)
        
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):
        return {
            'static': {k: v[idx] for k, v in self.static.items()},
            'sequence': self.sequence[idx],
            'id': self.ids[idx]
        }

class RecSysInference:
    def __init__(self, model, item_data_loader=None):
        """
        Args:
            model: Trained DualTowerRecModel.
            item_data_loader: DataLoader that yields all items (for offline indexing).
                              If None, index must be built manually later.
        """
        self.model = model
        self.model.eval()
        self.item_index = None
        self.item_ids = []
        self.use_faiss = False
        
        if item_data_loader:
            self.build_item_index(item_data_loader)
            
    def build_item_index(self, item_data_loader):
        """
        Pre-computes item embeddings and builds the index.
        """
        print("Building Item Index...")
        all_embeddings = []
        all_ids = []
        
        with torch.no_grad():
            for batch in item_data_loader:
                # batch is expected to be a dict with 'item_static', 'item_sequence', 'item_id'
                # Note: Our previous Dataset didn't return item_id explicitly in the dict structure 
                # suited for this, so we might need to adjust or assume the loader returns it.
                # For this implementation, let's assume the loader returns (features, ids) or similar.
                # Adjusting to match the structure we likely have:
                
                # Assuming batch is just the item features part of TripletRecDataset or similar
                # We need a dedicated ItemDataset for this.
                
                static = batch['static']
                sequence = batch['sequence']
                ids = batch['id'] # Expecting item IDs here
                
                embeddings = self.model.item_tower(static, sequence)
                # Normalize for Cosine Similarity
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_ids.extend(ids.numpy())
                
        self.item_embeddings = np.concatenate(all_embeddings, axis=0)
        self.item_ids = np.array(all_ids)
        
        if faiss:
            print("Using FAISS for indexing.")
            dim = self.item_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim) # Inner Product (Cosine since normalized)
            self.index.add(self.item_embeddings)
            self.use_faiss = True
        else:
            print("FAISS not found. Using PyTorch for indexing.")
            self.item_embeddings_tensor = torch.tensor(self.item_embeddings)
            self.use_faiss = False
            
        print(f"Index built for {len(self.item_ids)} items.")

    def get_recommendations(self, user_features, k=5) -> List[Tuple[int, float]]:
        """
        Args:
            user_features: Dict with 'categorical' and 'numerical' keys.
            k: Number of recommendations.
            
        Returns:
            List of (Item_ID, Score).
        """
        self.model.eval()
        with torch.no_grad():
            # Generate User Vector
            u_vec = self.model.user_tower(user_features['categorical'], user_features['numerical'])
            u_vec = torch.nn.functional.normalize(u_vec, p=2, dim=1)
            
            if self.use_faiss:
                # FAISS Search
                u_vec_np = u_vec.cpu().numpy()
                scores, indices = self.index.search(u_vec_np, k)
                
                # Map indices back to Item IDs
                # indices is (1, k)
                top_k_ids = self.item_ids[indices[0]]
                top_k_scores = scores[0]
                
            else:
                # PyTorch Search
                # (1, Dim) @ (N, Dim).T -> (1, N)
                scores = torch.matmul(u_vec, self.item_embeddings_tensor.t())
                top_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
                
                top_k_ids = self.item_ids[top_k_indices[0].numpy()]
                top_k_scores = top_k_scores[0].numpy()
                
        return list(zip(top_k_ids, top_k_scores))

# --- Verification ---
if __name__ == "__main__":
    from recsys_model import DualTowerRecModel
    
    # 2. Setup
    item_dataset = ItemDataset(num_items=50)
    item_loader = DataLoader(item_dataset, batch_size=10)
    
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
    
    # 3. Initialize Inference Engine
    inference_engine = RecSysInference(model, item_loader)
    
    # 4. Get Recommendations
    print("\nGenerating Recommendations...")
    mock_user = {
        'categorical': {
            'risk_preference': torch.tensor([[2]]), # Aggressive
            'investment_goal': torch.tensor([[0]])  # Growth
        },
        'numerical': torch.tensor([[0.5, 0.8]]) # Normalized Age, Income
    }
    
    recs = inference_engine.get_recommendations(mock_user, k=5)
    print("Top-5 Recommendations (ID, Score):")
    for item_id, score in recs:
        print(f"Item {item_id}: {score:.4f}")
