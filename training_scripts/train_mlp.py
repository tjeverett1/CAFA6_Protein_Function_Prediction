import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Import our new Dataset class
from training_scripts.dataset import ProteinEnsembleDataset

# ==========================================
# 1. MODEL
# ==========================================
class ProteinFunctionMLP(nn.Module):
    def __init__(self, feature_dim, num_classes, num_taxonomies, tax_emb_dim=32, hidden_dim=512, dropout_rate=0.3):
        super(ProteinFunctionMLP, self).__init__()
        
        # Taxonomy Embedding Layer
        self.tax_embedding = nn.Embedding(num_taxonomies, tax_emb_dim)
        
        # Total Input Dimension = (ESM + T5) + Tax_Emb
        total_input_dim = feature_dim + tax_emb_dim
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(total_input_dim, hidden_dim),
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

    def forward(self, features, tax_idx):
        # 1. Embed Taxonomy: (Batch) -> (Batch, Tax_Emb_Dim)
        tax_vec = self.tax_embedding(tax_idx)
        
        # 2. Concatenate: (Batch, Feat_Dim) + (Batch, Tax_Emb_Dim) -> (Batch, Total_Dim)
        x = torch.cat([features, tax_vec], dim=1)
        
        return self.network(x)

# ==========================================
# 2. TRAINER CLASS
# ==========================================
class ProteinTrainer:
    def __init__(self, config):
        self.config = config
        
        # Device Selection
        self.device = config.get("device")
        if self.device is None:
             if torch.cuda.is_available(): self.device = "cuda"
             elif torch.backends.mps.is_available(): self.device = "mps"
             else: self.device = "cpu"
        print(f"🔧 Using device: {self.device}")

        # --- Data Loading ---
        pickle_path = config['pickle_path']
        t5_pickle_path = config['t5_pickle_path']
        val_fold = config.get('val_fold', 0)
        
        # Initialize Datasets
        print("📦 Initializing Datasets...")
        self.train_dataset = ProteinEnsembleDataset(pickle_path, t5_pickle_path, mode='train', val_fold=val_fold)
        self.val_dataset = ProteinEnsembleDataset(pickle_path, t5_pickle_path, mode='val', val_fold=val_fold)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        
        # --- Model Setup ---
        # Feature dim = ESM(1280) + T5(1024) = 2304 (Adjust if different)
        feature_dim = config.get('feature_dim', 2304) 
        num_taxonomies = self.train_dataset.num_taxonomies
        
        self.model = ProteinFunctionMLP(
            feature_dim=feature_dim,
            num_classes=config['num_classes'],
            num_taxonomies=num_taxonomies,
            hidden_dim=config['hidden_dim'],
            dropout_rate=config.get('dropout', 0.3)
        ).to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3
        )

        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx+1}", leave=False)
        
        for batch in pbar:
            # Unpack batch
            features = batch['features'].to(self.device)
            tax_idx = batch['taxonomy_idx'].to(self.device)
            targets = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features, tax_idx)
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
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                tax_idx = batch['taxonomy_idx'].to(self.device)
                targets = batch['label'].to(self.device)
                
                outputs = self.model(features, tax_idx)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
        return running_loss / len(self.val_loader)

    def plot_live(self):
        try:
            clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            plt.plot(self.history['train_loss'], label='Train Loss', marker='o')
            plt.plot(self.history['val_loss'], label='Val Loss', marker='x')
            plt.title(f"Training Progress (Fold {self.config.get('val_fold',0)} | LR: {self.optimizer.param_groups[0]['lr']:.1e})")
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.legend()
            plt.grid(True)
            plt.show()
        except:
            pass

    def run(self):
        print(f"🚀 Starting Run | Hidden: {self.config['hidden_dim']} | LR: {self.config['learning_rate']}")
        
        for epoch in range(self.config['epochs']):
            t_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(t_loss)
            
            v_loss = self.validate()
            self.history['val_loss'].append(v_loss)
            
            self.scheduler.step(v_loss)
            
            if v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                # Save with config details
                save_name = f"best_model_h{self.config['hidden_dim']}_lr{self.config['learning_rate']}_fold{self.config.get('val_fold',0)}.pth"
                torch.save(self.model.state_dict(), save_name)

            self.plot_live()
            print(f"Epoch {epoch+1}/{self.config['epochs']} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")

        return self.best_val_loss

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Example config for testing
    # You must upload 'protein_data.pkl' to data/ before running this
    config = {
        "pickle_path": "data/protein_data.pkl",
        "t5_pickle_path": "data/t5_data.pkl",
        "feature_dim": 2304,    # 1280 (ESM) + 1024 (T5)
        "num_classes": 1024,
        "hidden_dim": 512,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 5,
        "val_fold": 0  # Use fold 0 for validation
    }
    # trainer = ProteinTrainer(config)
    # trainer.run()
