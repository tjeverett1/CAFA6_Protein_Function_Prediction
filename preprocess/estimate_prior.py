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
    "vocab_path": "data/labels_top1024.npy", 
    "num_classes": 1024,
    "subsample_size": 10000 
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
    
    # Optional: Subsample for speed
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
    
    # Normalize X (StandardScaler style) for better convergence
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6 # Avoid div by zero
    X = (X - mean) / std
    
    # Debug: Check if we have any positives
    Y = np.array(Y, dtype=np.float32)
    print(f"DEBUG: Y shape: {Y.shape}, Y sum: {Y.sum()}")
    
    if Y.sum() == 0:
        print("❌ WARNING: Target matrix Y is still all zeros!")
        if len(ids) > 0:
             print(f"Sample raw label: {data_dict[ids[0]]['labels']}")

    # Y might be a list of arrays, so we stack them ensuring they are 2D
    if Y.ndim == 1:
        try:
            Y = np.vstack(Y)
        except:
            print(f"❌ Error stacking Y. Shape before stack: {Y.shape}")
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
            
        # Skip extremely rare classes (unstable estimation)
        if y_c.sum() < 5:
             priors[c] = y_c.mean()
             continue
            
        # Split into a hold-out set for estimating 'c'
        # We train on a subset, estimate on the hold-out P set
        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y_c, test_size=0.2, stratify=y_c, random_state=42)
        
        # Train classifier g(x) to predict s=1 vs s=0
        # Logistic Regression is fast and works well for this
        clf = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced', n_jobs=1)
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

