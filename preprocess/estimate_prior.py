import numpy as np
import pickle
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "pickle_path": "data/protein_data.pkl",
    "t5_pickle_path": "data/t5_data.pkl",
    "output_prior_path": "data/class_priors.npy",
    "num_classes": 1024,
    "subsample_size": 10000 # Downsample for speed during prior estimation if needed
}

def load_data():
    print(f"📦 Loading data for prior estimation...")
    with open(CONFIG["pickle_path"], "rb") as f:
        data_dict = pickle.load(f)
    
    # We need features (X) and labels (Y)
    # For prior estimation, a subset of features (e.g. just ESM) is usually sufficient and faster
    # to distinguish P vs U distributions.
    
    ids = list(data_dict.keys())
    
    # Optional: Subsample for speed
    if len(ids) > CONFIG["subsample_size"]:
        np.random.seed(42)
        ids = np.random.choice(ids, CONFIG["subsample_size"], replace=False)
    
    X = []
    Y = []
    
    for pid in ids:
        item = data_dict[pid]
        # Using 'embedding' key (ESM)
        X.append(item['embedding'])
        
        raw_label = item['labels']
        
        # Handle sparse dictionary format for labels if necessary
        if isinstance(raw_label, dict):
            # Assuming dict keys are indices of positive labels
            dense_label = np.zeros(CONFIG["num_classes"], dtype=np.float32)
            for idx in raw_label:
                if isinstance(idx, int) and 0 <= idx < CONFIG["num_classes"]:
                    dense_label[idx] = 1.0
            Y.append(dense_label)
        elif hasattr(raw_label, "toarray"): # Scipy sparse matrix
             Y.append(raw_label.toarray().flatten())
        elif isinstance(raw_label, (list, np.ndarray, tuple)):
            # Check if it's a dense vector or a list of indices
            raw_label_arr = np.array(raw_label)
            if raw_label_arr.ndim == 1 and len(raw_label_arr) < CONFIG["num_classes"]:
                 # Likely a list of indices [1, 5, 100]
                 dense_label = np.zeros(CONFIG["num_classes"], dtype=np.float32)
                 for idx in raw_label_arr:
                     if int(idx) < CONFIG["num_classes"]:
                         dense_label[int(idx)] = 1.0
                 Y.append(dense_label)
            else:
                 # Assume it's already a dense vector
                 Y.append(raw_label)
        else:
            # Assume it's already a list/array
            Y.append(raw_label)
          
    # Ensure X and Y are proper 2D arrays
    X = np.array(X, dtype=np.float32)
    
    # Debug: Check if we have any positives
    Y = np.array(Y, dtype=np.float32)
    print(f"DEBUG: Y shape: {Y.shape}, Y sum: {Y.sum()}")
    if Y.sum() == 0:
        print("❌ WARNING: Target matrix Y is all zeros! Check label parsing.")
    
    if Y.ndim == 1:
    
    # Y might be a list of arrays, so we stack them ensuring they are 2D
    # If item['labels'] is already an array, np.array(Y) usually works,
    # but np.stack or np.vstack is safer if they are uniform.
    Y = np.array(Y, dtype=np.float32)
    
    if Y.ndim == 1:
        # This happens if Y is an array of objects (arrays), need to stack manually if np.array failed to infer dims
        # or if it's just a single sample (unlikely here)
        try:
            Y = np.vstack(Y)
        except:
            print(f"❌ Error stacking Y. Shape before stack: {Y.shape}")
            # Debug first element
            print(f"First element type: {type(Y[0])}, shape: {getattr(Y[0], 'shape', 'N/A')}")
            raise

    return X, Y

def estimate_priors(X, Y_multi):
    """
    Estimates prior pi_p for each class using Elkan-Noto method.
    """
    num_classes = Y_multi.shape[1]
    priors = np.zeros(num_classes)
    
    print(f"📊 Estimating priors for {num_classes} classes using Elkan-Noto...")
    
    # Iterate over each class individually
    for c in tqdm(range(num_classes)):
        y_c = Y_multi[:, c] # Labels for class c (1=Positive, 0=Unlabeled)
        
        # If no positives, prior is 0
        if y_c.sum() == 0:
            priors[c] = 0.0
            continue
            
        # Split into a hold-out set for estimating 'c'
        # We train on a subset, estimate on the hold-out P set
        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y_c, test_size=0.2, stratify=y_c, random_state=42)
        
        # Train classifier g(x) to predict s=1 vs s=0
        # Logistic Regression is fast and works well for this
        clf = LogisticRegression(solver='lbfgs', max_iter=200, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        # Estimate c = P(s=1 | y=1)
        # We look at the hold-out set where we KNOW labels are 1
        positives_holdout = X_holdout[y_holdout == 1]
        
        if len(positives_holdout) == 0:
             # Fallback if split resulted in no positives (rare)
             priors[c] = y_c.mean() # Naive estimate
             continue

        # Predict probabilities P(s=1|x) for true positives
        preds_pos = clf.predict_proba(positives_holdout)[:, 1]
        
        # c is the average probability assigned to true positives
        c_hat = np.mean(preds_pos)
        
        # Sanity clip
        c_hat = max(c_hat, 1e-5)
        
        # Calculate pi_p = P(s=1) / c
        # P(s=1) is just the fraction of labeled examples in dataset
        p_s1 = y_c.mean()
        
        pi_p = p_s1 / c_hat
        
        # Clip to [0, 1] range
        priors[c] = np.clip(pi_p, 0.0, 1.0)
        
    return priors

if __name__ == "__main__":
    X, Y = load_data()
    priors = estimate_priors(X, Y)
    
    print(f"💾 Saving priors to {CONFIG['output_prior_path']}")
    np.save(CONFIG['output_prior_path'], priors)
    
    print(f"Stats: Min={priors.min():.4f}, Max={priors.max():.4f}, Mean={priors.mean():.4f}")

