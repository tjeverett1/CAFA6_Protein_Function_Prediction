import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ==========================================
# 1. DATASET
# ==========================================
class ProteinEmbeddingDataset(Dataset):
    def __init__(self, ids_path, labels_path, emb_dir, input_dim):
        self.ids = np.load(ids_path)
        self.labels = np.load(labels_path)
        self.emb_dir = emb_dir
        self.input_dim = input_dim
        
        assert len(self.ids) == len(self.labels), "IDs and Labels count mismatch!"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        emb_path = os.path.join(self.emb_dir, f"{pid}.npz")
        
        try:
            data = np.load(emb_path)
            embedding = data['embedding']
        except Exception as e:
            # print(f"Error loading {pid}: {e}") # Silence for speed in loops
            embedding = np.zeros(self.input_dim)

        label = self.labels[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ==========================================
# 2. MODEL
# ==========================================
class ProteinFunctionMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, dropout_rate=0.3):
        super(ProteinFunctionMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output Layer
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. TRAINER CLASS
# ==========================================
class ProteinTrainer:
    def __init__(self, config):
        """
        config: dict with keys like 'hidden_dim', 'learning_rate', 'batch_size', etc.
        """
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare Data
        self.full_dataset = ProteinEmbeddingDataset(
            config['ids_path'], 
            config['labels_path'], 
            config['embedding_dir'],
            config['input_dim']
        )
        
        # 80/20 Split
        train_size = int(0.8 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        
        # Model setup
        self.model = ProteinFunctionMLP(
            config['input_dim'], 
            config['num_classes'], 
            config['hidden_dim'],
            config.get('dropout', 0.3)
        ).to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        # Scheduler: Reduce LR if Val Loss stops improving
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )

        # Tracking
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx+1}", leave=False)
        
        for i, (embeddings, targets) in enumerate(pbar):
            embeddings, targets = embeddings.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for embeddings, targets in self.val_loader:
                embeddings, targets = embeddings.to(self.device), targets.to(self.device)
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
        return running_loss / len(self.val_loader)

    def plot_live(self):
        """Updates the plot in the Notebook output."""
        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(self.history['val_loss'], label='Val Loss', marker='x')
        plt.title(f"Training Progress (LR: {self.optimizer.param_groups[0]['lr']:.1e})")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        print(f"🚀 Starting Run | Hidden: {self.config['hidden_dim']} | LR: {self.config['learning_rate']}")
        
        for epoch in range(self.config['epochs']):
            # Train
            t_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(t_loss)
            
            # Validate
            v_loss = self.validate()
            self.history['val_loss'].append(v_loss)
            
            # Scheduler Step
            self.scheduler.step(v_loss)
            
            # Checkpointing
            if v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                save_path = f"best_model_h{self.config['hidden_dim']}_lr{self.config['learning_rate']}.pth"
                torch.save(self.model.state_dict(), save_path)
                # print(f"⭐ New Best Val Loss: {v_loss:.4f} -> Saved {save_path}")

            # Live Plot
            self.plot_live()
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")

        return self.best_val_loss

# ==========================================
# 4. MAIN EXECUTION (Local Test)
# ==========================================
if __name__ == "__main__":
    # Example usage for testing locally
    local_config = {
        "embedding_dir": "data/train_embeddings", 
        "ids_path": "data/train_ids.npy",
        "labels_path": "data/train_targets_top1024.npy",
        "input_dim": 1280,
        "num_classes": 1024,
        "hidden_dim": 512,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 5,
        "device": "cpu" # Force CPU for quick local test
    }
    # trainer = ProteinTrainer(local_config)
    # trainer.run()
