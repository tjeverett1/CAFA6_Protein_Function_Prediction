import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import f1_score
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
# Import our new Dataset class
# from dataset import ProteinEnsembleDataset
from training_scripts.dataset import ProteinEnsembleDataset




timestamp = datetime.now().strftime("%m%d_%H%M")


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

class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.bad_epochs = 0

    def step(self, value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False  # don't stop if metric invalid
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

class TaxonomyLineageEncoder(nn.Module):
    def __init__(self, num_nodes, emb_dim=32, num_ranks=8, rank_emb_dim=8):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.rank_emb = nn.Embedding(num_ranks, rank_emb_dim)
        self.proj = nn.Linear(emb_dim + rank_emb_dim, emb_dim)

    def forward(self, lineage_idx):  # [B, R]
        B, R = lineage_idx.shape
        node_vecs = self.node_emb(lineage_idx)  # [B, R, emb_dim]

        ranks = torch.arange(R, device=lineage_idx.device)
        rank_vecs = self.rank_emb(ranks)[None, :, :]  # [1, R, rank_emb_dim]
        rank_vecs = rank_vecs.expand(B, R, -1)

        x = torch.cat([node_vecs, rank_vecs], dim=-1)  # [B, R, emb_dim+rank]
        x = self.proj(x)                               # [B, R, emb_dim]
        return x.sum(dim=1)                            # [B, emb_dim]

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class ProteinFunctionMLP(nn.Module):
    def __init__(self, feature_dim, num_classes, num_tax_nodes, num_ranks=8,
                 tax_emb_dim=32, hidden_dim=512, dropout_rate=0.3):
        super().__init__()

        self.tax_encoder = TaxonomyLineageEncoder(
            num_nodes=num_tax_nodes,
            emb_dim=tax_emb_dim,
            num_ranks=num_ranks
        )

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

    def forward(self, features, lineage_idx):
        tax_vec = self.tax_encoder(lineage_idx)  # [B, tax_emb_dim]
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
        vocab_path = config.get('vocab_path', 'data/labels_top1024.npy')
        val_fold = config.get('val_fold', 0)
        
        print("📦 Initializing Datasets...")
        
        if val_fold == -1:
            print("🔀 Random Split Mode (val_fold = -1)")
            # Load data to get IDs
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            all_ids = list(data.keys())
            
            # Deterministic shuffle
            random.seed(42)
            random.shuffle(all_ids)
            
            # 80/20 Split
            split_idx = int(len(all_ids) * 0.8)
            train_ids = all_ids[:split_idx]
            val_ids = all_ids[split_idx:]
            
            self.train_dataset = ProteinEnsembleDataset(pickle_path, t5_pickle_path, vocab_path, mode='train', val_fold=val_fold, specific_ids=train_ids)
            self.val_dataset = ProteinEnsembleDataset(pickle_path, t5_pickle_path, vocab_path, mode='val', val_fold=val_fold, specific_ids=val_ids)
        else:
            self.train_dataset = ProteinEnsembleDataset(pickle_path, t5_pickle_path, vocab_path, mode='train', val_fold=val_fold)
            self.val_dataset = ProteinEnsembleDataset(pickle_path, t5_pickle_path, vocab_path, mode='val', val_fold=val_fold)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        self.early_stopper = EarlyStopper(
            patience=self.config.get("early_stop_patience", 6),
            min_delta=self.config.get("early_stop_min_delta", 1e-3)
        )

        
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
            num_tax_nodes=self.train_dataset.num_tax_nodes,
            num_ranks=self.train_dataset.num_ranks,
            hidden_dim=config['hidden_dim'],
            dropout_rate=config.get('dropout', 0.3)
        ).to(self.device)
        
        # SWAP TO nnPU LOSS
        self.criterion = nnPULoss(priors=self.priors)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_pos_recall': [],
            'val_pred_pos_rate': [],
            'val_f1_labeled': []
        }
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0

        num_batches = len(self.train_loader)
        k = self.config.get("f1_subsample_batches", 0)
        f1_batches = set(np.random.choice(num_batches, size=min(k, num_batches), replace=False))

        subsample_preds = []
        subsample_targets = []

        for step, batch in enumerate(self.train_loader):
            features = batch['features'].to(self.device)
            targets = batch['label'].to(self.device)
            lineage_idx = batch['lineage_idx'].to(self.device)  # [B, R]
            


            self.optimizer.zero_grad()
            outputs = self.model(features, lineage_idx)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if step in f1_batches:
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                    subsample_preds.append(preds)
                    subsample_targets.append(targets.cpu().numpy())

        # train_f1 = None
        # if subsample_preds:
        #     train_f1 = f1_score(
        #         np.vstack(subsample_targets),
        #         np.vstack(subsample_preds),
        #         average="micro"
        #     )

        return running_loss / len(self.train_loader) #, train_f1



    def validate(self):
        self.model.eval()
        running_loss = 0.0

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                lineage_idx = batch['lineage_idx'].to(self.device)  # [B, R]
                targets = batch['label'].to(self.device)

                logits = self.model(features, lineage_idx)
                loss = self.criterion(logits, targets)
                running_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_targets.append(targets.cpu())

        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        # ---------- PU-SAFE METRICS ----------
        preds = (probs > 0.5).astype(np.int32)

        # 1) Recall on labeled positives ONLY
        pos_mask = (targets == 1)
        if pos_mask.sum() > 0:
            pos_recall = (preds[pos_mask] == 1).mean()
        else:
            pos_recall = np.nan

        # 2) Diagnostic: predicted positive rate
        pred_pos_rate = preds.mean()

        # 3) Optional: micro-F1 but ONLY over labeled positives
        try:
            f1_labeled = f1_score(
                targets[pos_mask],
                preds[pos_mask],
                average="micro"
            )
        except:
            f1_labeled = np.nan

        metrics = {
            "loss": running_loss / len(self.val_loader),
            "pos_recall": pos_recall,
            "pred_pos_rate": pred_pos_rate,
            "f1_labeled": f1_labeled
        }

        return metrics


    def plot_live(self):
        try:
            clear_output(wait=True)
            fig, ax1 = plt.subplots(figsize=(10, 5))

            # Loss
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('nnPU Loss', color='tab:blue')
            ax1.plot(self.history['train_loss'], label='Train Loss', marker='o', alpha=0.6)
            ax1.plot(self.history['val_loss'], label='Val Loss', marker='x')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # PU-safe metrics
            ax2 = ax1.twinx()
            ax2.set_ylabel('PU Metrics', color='tab:orange')
            ax2.plot(self.history['val_pos_recall'], label='Val Pos Recall', marker='s')
            ax2.plot(self.history['val_pred_pos_rate'], label='Pred + Rate', marker='.')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.legend(loc='upper right')

            plt.title(f"nnPU Training (Fold {self.config.get('val_fold', 0)})")
            fig.tight_layout()
            plt.show()
        except:
            pass


    def run(self):
        print(f"🚀 Starting Run | Hidden: {self.config['hidden_dim']} | LR: {self.config['learning_rate']}")
        
        for epoch in range(self.config['epochs']):
            t_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(t_loss)
            
            
            val_metrics = self.validate()

            self.history['val_loss'].append(val_metrics["loss"])
            self.history['val_pos_recall'].append(val_metrics["pos_recall"])
            self.history['val_pred_pos_rate'].append(val_metrics["pred_pos_rate"])
            self.history['val_f1_labeled'].append(val_metrics["f1_labeled"])

            self.scheduler.step(val_metrics["loss"])

            print(
                f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"Train Loss: {t_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Pos Recall: {val_metrics['pos_recall']:.4f} | "
                f"Pred+ Rate: {val_metrics['pred_pos_rate']:.4f}"
            )
            
            
            
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                save_name = (
                    f"models/kfold5/best_model_"
                    f"h{self.config['hidden_dim']}_"
                    f"lr{self.config['learning_rate']}_"
                    f"fold{self.config.get('val_fold', 0)}_"
                    f"{timestamp}.pth"
                )
                torch.save(
                    {
                        "state_dict": self.model.state_dict(),
                        "num_taxonomies": self.train_dataset.num_taxonomies,
                        "taxonomy_mapping": self.train_dataset.tax_to_idx,
                        "config": self.config,
                    },
                    save_name
                )
                # Early stop on PU-safe metric

            if self.early_stopper.step(val_metrics["pos_recall"]):
                print(f"🛑 Early stopping: pos_recall stopped improving (best={self.early_stopper.best:.4f})")
                break


            # self.plot_live()
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"Train Loss: {t_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Pos Recall: {val_metrics['pos_recall']:.4f} | "
                f"Pred+ Rate: {val_metrics['pred_pos_rate']:.4f}"
            )

        return self.best_val_loss
    def load_model(self, path):
        print(f"📥 Loading model from {path}")
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    
# def inference(self, save_path="val_predictions.npy"):
#     self.model.eval()
#     all_preds = []
#     all_targets = []
#     all_ids = []

#     with torch.no_grad():
#         for batch in tqdm(self.val_loader, desc="Running inference"):
#             features = batch['features'].to(self.device)
#             lineage_idx = batch['lineage_idx'].to(self.device)  # [B, R]
#             targets = batch['label'].cpu().numpy()  # save ground truth
#             ids = batch['id'] if 'id' in batch else None

#             logits = self.model(features, lineage_idx)
#             probs = torch.sigmoid(logits).cpu().numpy()
#             preds = (probs > 0.5).astype(np.float32)

#             all_preds.append(preds)
#             all_targets.append(targets)
#             if ids is not None:
#                 all_ids += list(ids)

#     all_preds = np.vstack(all_preds)
#     all_targets = np.vstack(all_targets)

#     # Save output dictionary
#     output = {
#         "ids": all_ids,
#         "pred_probs": all_preds,
#         "targets": all_targets
#     }

#     np.save(save_path, output)
#     print(f"💾 Saved inference results → {save_path}")

#     # Optional: Compute F1
#     try:
#         val_f1 = f1_score(all_targets, (all_preds > 0.5), average="micro")
#         print(f"📊 Inference Micro F1 = {val_f1:.4f}")
#     except Exception as e:
#         print("Could not compute F1:", e)

#     return output

def run_kfold_training(base_config, folds=(0, 1, 2, 3, 4)):
    fold_results = {}

    # Make sure models/ exists
    os.makedirs("models", exist_ok=True)

    for fold in folds:
        print("\n" + "=" * 80)
        print(f"🚀 Training fold {fold} (val_fold={fold})")
        print("=" * 80)

        cfg = dict(base_config)
        cfg["val_fold"] = fold

        trainer = ProteinTrainer(cfg)
        best_val = trainer.run()

        fold_results[fold] = {
            "best_val_loss": best_val,
            "final_val_loss": trainer.history["val_loss"][-1] if trainer.history["val_loss"] else None,
            "final_pos_recall": trainer.history.get("val_pos_recall", [None])[-1],
            "final_pred_pos_rate": trainer.history.get("val_pred_pos_rate", [None])[-1],
        }

    print("\n✅ K-fold training complete. Summary:")
    for fold, res in fold_results.items():
        print(f"Fold {fold}: best_val_loss={res['best_val_loss']:.4f} | "
              f"final_pos_recall={res['final_pos_recall']}")
    return fold_results



# ==========================================
# 4. MAIN EXECUTION (Local Test)
# ==========================================
if __name__ == "__main__":
    base_config = {
        "pickle_path": "data/protein_data.pkl",
        "t5_pickle_path": "data/t5_data.pkl",
        "prior_path": "data/class_priors.npy",
        "vocab_path": "data/labels_top1024.npy",

        # keep if you still use it elsewhere; otherwise safe to leave
        "taxonomy_cache_path": "data/taxonomy_mapping.pkl",

        "f1_subsample_batches": 10,
        # "feature_dim": 2304,
        "feature_dim": 1024, # T5 only
        "num_classes": 1024,
        "hidden_dim": 512,
        "batch_size": 512,
        "learning_rate": 7e-4,
        "epochs": 30,           
        "early_stop_patience": 2,
        "early_stop_min_delta": 7e-4,

        # will be overwritten inside kfold loop
        "val_fold": 0,
    }

    results = run_kfold_training(base_config, folds=(0, 1, 2, 3, 4))
    # trainer = ProteinTrainer(base_config)
#     trainer.run()

    # Optional: save fold summary
    with open("models/kfold_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("💾 Saved fold results → models/kfold_results.pkl")

# if __name__ == "__main__":
#     config = {
#         "pickle_path": "data/protein_data.pkl",
#         "t5_pickle_path": "data/t5_data.pkl",
#         "prior_path": "data/class_priors.npy",
#         'vocab_path': 'data/labels_top1024.npy',
#         'taxonomy_cache_path': 'data/taxonomy_mapping.pkl',
#         "f1_subsample_batches": 10 ,  # ~10 batches per epoch
#         "feature_dim": 2304,
#         "num_classes": 1024,
#         "hidden_dim": 512,
#         "batch_size": 512,
#         "learning_rate": 1e-3,
#         "epochs": 5,
#         "val_fold": 0 
#     }
#     trainer = ProteinTrainer(config)
#     trainer.run()
    
#     # LOAD A SAVED MODEL
#     # trainer.load_model("BASELINE.pth")

#     # RUN INFERENCE
#     # results = trainer.inference(save_path="val_predictions.npy")

