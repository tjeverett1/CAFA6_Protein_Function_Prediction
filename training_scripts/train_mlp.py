import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    # Adjust this path to where your .npz files actually are
    "embedding_dir": "cafa-6-protein-function-prediction/train_embeddings", 
    "ids_path": "cafa-6-protein-function-prediction/Train/train_ids.npy",  # Output from generate_go_labels.py
    "labels_path": "train_targets_top1024.npy",                            # Output from generate_go_labels.py
    "input_dim": 1280,      # ESM-2 t33 dimension
    "num_classes": 1024,    # Number of GO terms
    "hidden_dim": 512,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
}

# ==========================================
# 2. CUSTOM DATASET
# ==========================================
class ProteinEmbeddingDataset(Dataset):
    def __init__(self, ids_path, labels_path, emb_dir):
        # Load the "Map" (IDs) and the "Answers" (Labels)
        self.ids = np.load(ids_path)
        self.labels = np.load(labels_path)
        self.emb_dir = emb_dir
        
        # Sanity check
        assert len(self.ids) == len(self.labels), "IDs and Labels count mismatch!"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        
        # Load the specific embedding file for this protein
        # explore_data.py saves as compressed .npz with key 'embedding'
        emb_path = os.path.join(self.emb_dir, f"{pid}.npz")
        
        try:
            # Load .npz
            data = np.load(emb_path)
            embedding = data['embedding'] # Shape (1280,)
        except Exception as e:
            # Fallback if file missing/corrupt
            print(f"Error loading {pid}: {e}")
            embedding = np.zeros(CONFIG['input_dim'])

        # Get label vector
        label = self.labels[idx]

        # Return as Float Tensors
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ==========================================
# 3. MODEL DEFINITION (MLP)
# ==========================================
class ProteinFunctionMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim):
        super(ProteinFunctionMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output Layer
            nn.Linear(hidden_dim // 2, num_classes)
            # Note: No Sigmoid here because we use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    print(f"🚀 Training on {CONFIG['device']}")
    
    # --- Prepare Data ---
    print("📦 Loading Dataset...")
    try:
        full_dataset = ProteinEmbeddingDataset(
            CONFIG['ids_path'], 
            CONFIG['labels_path'], 
            CONFIG['embedding_dir']
        )
    except FileNotFoundError as e:
        print(f"❌ Could not find data files: {e}")
        print("Make sure you ran 'generate_go_labels.py' first!")
        return
    
    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # --- Initialize Model ---
    model = ProteinFunctionMLP(CONFIG['input_dim'], CONFIG['num_classes'], CONFIG['hidden_dim'])
    model = model.to(CONFIG['device'])
    
    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # --- Loop ---
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for embeddings, targets in progress_bar:
            embeddings, targets = embeddings.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            # Forward
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        # --- Validation Step ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for embeddings, targets in val_loader:
                embeddings, targets = embeddings.to(CONFIG['device']), targets.to(CONFIG['device'])
                outputs = model(embeddings)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    # --- Save Model ---
    torch.save(model.state_dict(), "mlp_model.pth")
    print("💾 Model saved to mlp_model.pth")

if __name__ == "__main__":
    train()



