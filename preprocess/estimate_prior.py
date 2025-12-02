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
    "vocab_path": "data/labels_top1024.npy", # Added vocab path
    "num_classes": 1024,
    "subsample_size": 10000 
}

def load_data():
    print(f"📦 Loading data for prior estimation...")
    
    # Load Vocabulary if available
    term_to_idx = {}
    if os.path.exists(CONFIG["vocab_path"]):
        vocab = np.load(CONFIG["vocab_path"])
        term_to_idx = {term: i for i, term in enumerate(vocab)}
        print(f"📖 Loaded vocabulary with {len(term_to_idx)} terms")
    else:
        print("⚠️ Warning: No vocabulary found at data/labels_top1024.npy. Assuming labels are already indices.")

    with open(CONFIG["pickle_path"], "rb") as f:
        data_dict = pickle.load(f)
    
    # We need features (X) and labels (Y)
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
        
        # DEBUG: Print first label structure
        if i == 0:
            print(f"\n🔍 DEBUG: First raw_label type: {type(raw_label)}")
            print(f"🔍 DEBUG: First raw_label content: {raw_label}")
        
        dense_label = np.zeros(CONFIG["num_classes"], dtype=np.float32)

        # Helper to process a single label item (int or string)
        def process_label_item(lbl, d_label):
            if isinstance(lbl, int):
                if 0 <= lbl < CONFIG["num_classes"]:
                    d_label[lbl] = 1.0
            elif isinstance(lbl, str):
                if lbl in term_to_idx:
                    d_label[term_to_idx[lbl]] = 1.0

        # Handle various formats
        if isinstance(raw_label, dict):
            for key in raw_label:
                process_label_item(key, dense_label)
        elif hasattr(raw_label, "toarray"): # Scipy sparse matrix
             dense_label = raw_label.toarray().flatten()[:CONFIG["num_classes"]]
        elif isinstance(raw_label, (list, np.ndarray, tuple)):
            raw_label_arr = np.array(raw_label)
            if np.issubdtype(raw_label_arr.dtype, np.number) and raw_label_arr.shape == (CONFIG["num_classes"],):
                 dense_label = raw_label_arr
            else:
                 for val in raw_label:
                     process_label_item(val, dense_label)
        else:
            process_label_item(raw_label, dense_label)
            
        Y.append(dense_label)
          
    # Ensure X and Y are proper 2D arrays
    X = np.array(X, dtype=np.float32)
    
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

