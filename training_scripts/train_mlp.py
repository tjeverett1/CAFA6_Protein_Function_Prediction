import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import f1_score

# Import our new Dataset class
from training_scripts.dataset import ProteinEnsembleDataset

# ==========================================
# 1. nnPU LOSS
# ==========================================
class nnPULoss(nn.Module):
    def __init__(self, priors, beta=0.0, gamma=1.0):
        """
        Non-Negative PU Loss.
        Args:
            priors (torch.Tensor): Prior probabilities pi_p for each class. Shape: (num_classes,)
            beta (float): Minimum value for negative risk (usually 0).
            gamma (float): Weight for the positive term (usually 1).
        """
        super(nnPULoss, self).__init__()
        self.priors = priors
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: (batch, num_classes) - Raw output from model
        targets: (batch, num_classes) - 0/1 labels
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Positive / Unlabeled masks
        # targets are 1 for P, 0 for U
        
        # Loss components
        # We use Binary Cross Entropy without reduction first to handle element-wise
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # But nnPU formula requires us to calculate specific terms:
        # g(x) is sigmoid(f(x))
        
        # L(g(x), 1) = -log(g(x))
        # L(g(x), 0) = -log(1 - g(x))
        
        # We can use BCE for this but we need to be careful with indices.
        # Since we are doing batch-wise operations for ALL classes at once:
        
        # P risk: E_p [ L(g(x), 1) ]
        # We approximate E_p by averaging over the Positive examples in the batch
        # But standard nnPU derivation usually assumes we have separate P and U datasets.
        # Here we have a mixed batch.
        
        # Let's implement the element-wise formulation:
        # For each class c:
        # R_p+ = (1/N_p) * Sum_{i in P} loss(g(xi), 1)
        # R_u- = (1/N_u) * Sum_{i in U} loss(g(xi), 0)
        # R_p- = (1/N_p) * Sum_{i in P} loss(g(xi), 0)  <-- Counterfactual loss
        
        # Masks
        is_p = (targets == 1).float()
        is_u = (targets == 0).float()
        
        n_p = is_p.sum(dim=0).clamp(min=1.0) # Number of positives per class in batch
        n_u = is_u.sum(dim=0).clamp(min=1.0) # Number of unlabeled per class in batch
        
        # Losses
        # loss_pos: L(g(x), 1)
        # loss_neg: L(g(x), 0)
        loss_pos_vec = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
        loss_neg_vec = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
        
        # Empirical Risks
        r_p_plus = (is_p * loss_pos_vec).sum(dim=0) / n_p
        r_u_minus = (is_u * loss_neg_vec).sum(dim=0) / n_u
        r_p_minus = (is_p * loss_neg_vec).sum(dim=0) / n_p
        
        # Unbiased Negative Risk Estimator
        # R_n_unbiased = R_u- - pi_p * R_p-
        # Note: R_u- estimates E_x[L(0)] - pi * E_p[L(0)] ... 
        # Actually the standard derivation for mixed U (which is P + N) is:
        # E_n[L(0)] = (E_u[L(0)] - pi * E_p[L(0)]) / (1 - pi)
        
        # However, standard nnPU implementations often simplify or assume U ~ p(x)
        # Let's stick to the formulation provided in the prompt:
        # Unbiased neg risk = R_u- - pi_p * R_p- 
        # (Assuming R_u- is calculated on the WHOLE U set which approximates p(x))
        
        # Correction: In our batch, "U" are samples labeled 0. 
        # If these are truly "Unlabeled" (mixture), then R_u_minus is correct.
        
        prior = self.priors.to(logits.device)
        
        neg_risk_unbiased = r_u_minus - prior * r_p_minus
        
        # Non-Negative Correction
        neg_risk = torch.clamp(neg_risk_unbiased, min=self.beta)
        
        # Total Loss per class
        # loss = pi_p * r_p_plus + neg_risk
        # We scale r_p_plus by prior? Or is r_p_plus already P(y=1|P)?
        # Standard risk: R(f) = pi * R_p+(f) + (1-pi) * R_n-(f)
        # So yes, we weight the positive risk by the prior.
        
        loss_per_class = prior * r_p_plus + neg_risk
        
        # Average over classes
        return loss_per_class.mean()


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class ProteinFunctionMLP(nn.Module):
    def __init__(self, feature_dim, num_classes, num_taxonomies, tax_emb_dim=32, hidden_dim=512, dropout_rate=0.3):
        super(ProteinFunctionMLP, self).__init__()
        self.tax_embedding = nn.Embedding(num_taxonomies, tax_emb_dim)
        total_input_dim = feature_dim + tax_emb_dim
        
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, features, tax_idx):
        tax_vec = self.tax_embedding(tax_idx)
        x = torch.cat([features, tax_vec], dim=1)
        return self.network(x)

# ==========================================
# 3. TRAINER CLASS
# ==========================================
class ProteinTrainer:
    def __init__(self, config):
        self.config = config
        
        self.device = config.get("device")
        if self.device is None:
             if torch.cuda.is_available(): self.device = "cuda"
             elif torch.backends.mps.is_available(): self.device = "mps"
             else: self.device = "cpu"
        print(f"🔧 Using device: {self.device}")

        # Load Data
        pickle_path = config['pickle_path']
        t5_pickle_path = config['t5_pickle_path']
        prior_path = config.get('prior_path', 'data/class_priors.npy') # Path to priors
        val_fold = config.get('val_fold', 0)
        
        print("📦 Initializing Datasets...")
        self.train_dataset = ProteinEnsembleDataset(pickle_path, t5_pickle_path, mode='train', val_fold=val_fold)
        self.val_dataset = ProteinEnsembleDataset(pickle_path, t5_pickle_path, mode='val', val_fold=val_fold)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        
        # Load Priors
        if os.path.exists(prior_path):
            print(f"✅ Loading Class Priors from {prior_path}")
            self.priors = torch.tensor(np.load(prior_path), dtype=torch.float32).to(self.device)
        else:
            print("⚠️ Priors file not found! Using naive default 0.01")
            self.priors = torch.full((config['num_classes'],), 0.01).to(self.device)

        # Model
        feature_dim = config.get('feature_dim', 2304) 
        self.model = ProteinFunctionMLP(
            feature_dim=feature_dim,
            num_classes=config['num_classes'],
            num_taxonomies=self.train_dataset.num_taxonomies,
            hidden_dim=config['hidden_dim'],
            dropout_rate=config.get('dropout', 0.3)
        ).to(self.device)
        
        # SWAP TO nnPU LOSS
        self.criterion = nnPULoss(priors=self.priors)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3
        )

        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx+1}", leave=False)
        
        for batch in pbar:
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
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                tax_idx = batch['taxonomy_idx'].to(self.device)
                targets = batch['label'].to(self.device)
                
                outputs = self.model(features, tax_idx)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        val_f1 = f1_score(all_targets, all_preds, average='micro')
        
        return running_loss / len(self.val_loader), val_f1

    def plot_live(self):
        try:
            clear_output(wait=True)
            fig, ax1 = plt.subplots(figsize=(10, 5))

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('nnPU Loss', color='tab:blue')
            ax1.plot(self.history['train_loss'], label='Train Loss', marker='o', color='tab:blue')
            ax1.plot(self.history['val_loss'], label='Val Loss', marker='x', color='tab:cyan')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Micro F1', color='tab:orange')
            ax2.plot(self.history['val_f1'], label='Val F1', marker='s', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            plt.title(f"nnPU Training (Fold {self.config.get('val_fold',0)})")
            fig.tight_layout()
            plt.show()
        except:
            pass

    def run(self):
        print(f"🚀 Starting Run | Hidden: {self.config['hidden_dim']} | LR: {self.config['learning_rate']}")
        
        for epoch in range(self.config['epochs']):
            t_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(t_loss)
            
            v_loss, v_f1 = self.validate()
            self.history['val_loss'].append(v_loss)
            self.history['val_f1'].append(v_f1)
            
            self.scheduler.step(v_loss)
            
            if v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                save_name = f"best_model_h{self.config['hidden_dim']}_lr{self.config['learning_rate']}_fold{self.config.get('val_fold',0)}.pth"
                torch.save(self.model.state_dict(), save_name)

            self.plot_live()
            print(f"Epoch {epoch+1}/{self.config['epochs']} | Train: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val F1: {v_f1:.4f}")

        return self.best_val_loss

# ==========================================
# 4. MAIN EXECUTION (Local Test)
# ==========================================
if __name__ == "__main__":
    config = {
        "pickle_path": "data/protein_data.pkl",
        "t5_pickle_path": "data/t5_data.pkl",
        "prior_path": "data/class_priors.npy",
        "feature_dim": 2304,
        "num_classes": 1024,
        "hidden_dim": 512,
        "batch_size": 2048,
        "learning_rate": 1e-3,
        "epochs": 5,
        "val_fold": 0
    }
    # trainer = ProteinTrainer(config)
    # trainer.run()
