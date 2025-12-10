import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "pickle_path": "data/protein_data.pkl",
    "t5_pickle_path": "data/t5_data.pkl",
    "output_prior_path": "data/class_priors.npy",
    "vocab_path": "data/labels_top1024.npy", 
    "num_classes": 1024,
    "subsample_size": 20000, # Increased subsample since GPU is fast
    "batch_size": 2048,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def build_vocab_from_data(data_dict, n_classes):
    print("⚠️ Vocabulary not found. Building from training data...")
    from collections import Counter
    counts = Counter()
    
    print("📊 Scanning data for top GO terms...")
    for pid in tqdm(data_dict):
        item = data_dict[pid]
        raw_label = item['labels']
        
        # Extract terms
        terms = []
        if isinstance(raw_label, dict):
            for val in raw_label.values():
                if isinstance(val, list): terms.extend(val)
                elif isinstance(val, str): terms.append(val)
        elif isinstance(raw_label, list):
            terms = raw_label
            
        counts.update(terms)
        
    top_terms = [t for t, c in counts.most_common(n_classes)]
    print(f"✔ Built vocabulary with {len(top_terms)} terms.")
    
    # Save it
    np.save(CONFIG["vocab_path"], np.array(top_terms))
    print(f"💾 Saved generated vocabulary to {CONFIG['vocab_path']}")
    
    return {term: i for i, term in enumerate(top_terms)}

def load_data():
    print(f"📦 Loading data for prior estimation...")

    with open(CONFIG["pickle_path"], "rb") as f:
        data_dict = pickle.load(f)

    # Load or Build Vocabulary
    term_to_idx = {}
    if os.path.exists(CONFIG["vocab_path"]):
        vocab = np.load(CONFIG["vocab_path"])
        term_to_idx = {term: i for i, term in enumerate(vocab)}
        print(f"📖 Loaded vocabulary with {len(term_to_idx)} terms")
    else:
        # Auto-generate if missing
        term_to_idx = build_vocab_from_data(data_dict, CONFIG["num_classes"])

    ids = list(data_dict.keys())
    
    # Optional: Subsample for speed (but GPU handles large data well)
    if len(ids) > CONFIG["subsample_size"]:
        np.random.seed(42)
        ids = np.random.choice(ids, CONFIG["subsample_size"], replace=False)
    
    X = []
    Y = []
    
    for i, pid in enumerate(ids):
        item = data_dict[pid]
        X.append(item['embedding'])
        
        raw_label = item['labels']
        dense_label = np.zeros(CONFIG["num_classes"], dtype=np.float32)

        # Extract all GO terms from nested structure
        current_terms = []
        if isinstance(raw_label, dict):
            for val in raw_label.values():
                if isinstance(val, list): current_terms.extend(val)
                elif isinstance(val, str): current_terms.append(val)
        elif isinstance(raw_label, list):
            current_terms = raw_label
        elif isinstance(raw_label, str):
             current_terms = [raw_label]

        # Map to indices
        for term in current_terms:
            if term in term_to_idx:
                dense_label[term_to_idx[term]] = 1.0
            
        Y.append(dense_label)
          
    # Ensure X and Y are proper 2D arrays
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    # Stack if needed
    if Y.ndim == 1 and len(Y) > 0:
         try: Y = np.vstack(Y)
         except: pass

    # Feature Scaling (Simple standardization)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X = (X - mean) / std

    print(f"DEBUG: Y shape: {Y.shape}, Y sum: {Y.sum()}")
    
    return X, Y

def estimate_priors_gpu(X, Y):
    """
    Estimates priors using a GPU-accelerated PyTorch model.
    Trains one multi-label classifier to predict P(s=1|x) for all classes simultaneously.
    """
    device = CONFIG["device"]
    print(f"🚀 Using device: {device}")
    
    # Convert to Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    
    # Split Train/Calib
    # We use a hold-out set (Calibration) to estimate 'c'
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    
    # Random shuffle indices
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    calib_idx = indices[n_train:]
    
    X_train, Y_train = X_tensor[train_idx], Y_tensor[train_idx]
    X_calib, Y_calib = X_tensor[calib_idx], Y_tensor[calib_idx]
    
    input_dim = X.shape[1]
    num_classes = Y.shape[1]
    
    # Simple Logistic Regression Model (Linear + Sigmoid)
    model = nn.Linear(input_dim, num_classes).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Weighted Loss to handle imbalance roughly
    # P(s=1) is usually low, so we can weight positives higher to help convergence
    # But for probability estimation, unweighted is often better calibrated.
    # Let's stick to standard BCE for probabilistic interpretation.
    criterion = nn.BCEWithLogitsLoss()
    
    print("🏋️ Training estimator model on GPU...")
    batch_size = CONFIG["batch_size"]
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        permutation = torch.randperm(n_train)
        
        epoch_loss = 0
        for i in range(0, n_train, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {epoch_loss:.4f}")

    # ---------------------------------------------------------
    # Estimate 'c' and Priors
    # ---------------------------------------------------------
    print("📊 Estimating priors from calibration set...")
    model.eval()
    with torch.no_grad():
        logits = model(X_calib)
        probs = torch.sigmoid(logits) # P(s=1|x)
    
    # Move to CPU for numpy ops
    probs = probs.cpu().numpy()
    Y_calib = Y_calib.cpu().numpy()
    
    priors = np.zeros(num_classes)
    
    for k in range(num_classes):
        # Positives in calibration set
        pos_mask = (Y_calib[:, k] == 1)
        
        # P(s=1) estimated from dataset frequency
        # Use the whole dataset frequency for stability
        p_s1 = Y[:, k].mean()
        
        if pos_mask.sum() < 5:
            # Too few positives to estimate c reliably
            # Assume c=1 (labeling is perfect) -> prior = p(s=1)
            priors[k] = p_s1
            continue
            
        # c = Average prediction on positives
        # c = P(s=1 | y=1)
        probs_pos = probs[pos_mask, k]
        c_hat = np.mean(probs_pos)
        
        # Clip c
        c_hat = max(c_hat, 0.01) 
        
        # pi_p = P(s=1) / c
        pi_p = p_s1 / c_hat
        
        priors[k] = np.clip(pi_p, 0.0, 1.0)
        
    return priors

if __name__ == "__main__":
    X, Y = load_data()
    priors = estimate_priors_gpu(X, Y)
    
    print(f"💾 Saving priors to {CONFIG['output_prior_path']}")
    np.save(CONFIG['output_prior_path'], priors)
    
    print(f"Stats: Min={priors.min():.4f}, Max={priors.max():.4f}, Mean={priors.mean():.4f}")

