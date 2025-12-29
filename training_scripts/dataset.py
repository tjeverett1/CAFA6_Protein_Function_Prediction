import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from collections import Counter

CONFIG = {
    'taxonomy_fasta_path': r'C:\Users\tessa\MIT Dropbox\Tessa Everett\6.s043\final_project\cafa-6-protein-function-prediction\Test\testsuperset.fasta',
    'taxonomy_tsv_path': r'C:\Users\tessa\MIT Dropbox\Tessa Everett\6.s043\final_project\cafa-6-protein-function-prediction\Test\testsuperset-taxon-list.tsv',
    'taxonomy_cache_path': 'data/taxonomy_mapping.pkl'
}

class ProteinEnsembleDataset(Dataset):
    def __init__(self, pickle_path, t5_pickle_path, vocab_path="data/labels_top1024.npy", mode='train', val_fold=0, specific_ids=None, top_k_taxonomies=1000):
        """
        Args:
            pickle_path (str): Path to the main dictionary (ESM + metadata).
            t5_pickle_path (str): Path to the T5 embeddings dictionary.
            taxonomy_path (str): Path to the taxonomy file.
            vocab_path (str): Path to the GO term vocabulary.
            mode (str): 'train' or 'val'.
            val_fold (int): The fold ID to use for validation.
            specific_ids (list, optional): If provided, use these IDs directly (ignoring folds).
        """
        print(f"📦 Loading main data from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            self.data_dict = pickle.load(f)
            
        print(f"📦 Loading T5 embeddings from {t5_pickle_path}...")
        with open(t5_pickle_path, "rb") as f:
            self.t5_dict = pickle.load(f)
            
        # Load Vocabulary
        self.top_k_taxonomies = top_k_taxonomies
        self.term_to_idx = {}
        if os.path.exists(vocab_path):
            vocab = np.load(vocab_path)
            self.term_to_idx = {term: i for i, term in enumerate(vocab)}
            print(f"📖 Loaded vocabulary with {len(self.term_to_idx)} terms")
            
        # Get list of all IDs
        all_ids = list(self.data_dict.keys())
        
        # 1. Build Taxonomy Map (Fixed across all folds)
        
        self.tax_to_idx, self.other_tax_idx, self.num_taxonomies =load_or_build_taxonomy_mapping(
                fasta_path=CONFIG['taxonomy_fasta_path'],
                tsv_path=CONFIG['taxonomy_tsv_path'],
                cache_path=CONFIG['taxonomy_cache_path'],
                top_k=top_k_taxonomies
            )
        

        # 2. Filter Data
        self.filtered_ids = []
        
        if specific_ids is not None:
            # Case A: Specific IDs provided (Random Split mode from Trainer)
            self.filtered_ids = specific_ids
            print(f"✔ {mode.upper()} set: {len(self.filtered_ids)} samples (Random Split)")
        else:
            # Case B: Fold-based Split
            unique_folds = set()
            for pid in all_ids:
                item = self.data_dict[pid]
                item_fold = item.get('fold') 
                unique_folds.add(item_fold)
                
                try:
                    item_fold = int(item_fold)
                    val_fold = int(val_fold)
                except (ValueError, TypeError):
                    pass 

                if mode == 'train':
                    if item_fold != val_fold:
                        self.filtered_ids.append(pid)
                elif mode == 'val':
                    if item_fold == val_fold:
                        self.filtered_ids.append(pid)
            
            print(f"✔ {mode.upper()} set: {len(self.filtered_ids)} samples (Val Fold: {val_fold}) | Folds present: {sorted(list(unique_folds))}")

    def __len__(self):
        return len(self.filtered_ids)

    def __getitem__(self, idx):
        pid = self.filtered_ids[idx]
        item = self.data_dict[pid]
        
        # 1. Features
        # ESM from main dict (key is 'embedding')
        esm = item['embedding']
        
        # T5 from separate dict (key is also 'embedding')
        # We access the item for this PID from the T5 dict
        t5_item = self.t5_dict.get(pid)
        
        if t5_item is not None:
             t5 = t5_item['embedding']
        else:
            # Fallback: zero vector of size 1024 (T5 standard)
            t5 = np.zeros(1024, dtype=np.float32)
        
        # Ensure they are numpy arrays
        if not isinstance(esm, np.ndarray): esm = np.array(esm)
        if not isinstance(t5, np.ndarray): t5 = np.array(t5)
            
        combined_features = np.concatenate([esm, t5])
        
        # 2. Taxonomy Index
        tax_raw = str(item.get("taxonomy"))
        tax_idx = self.tax_to_idx.get(tax_raw, self.other_tax_idx)
        assert len(self.tax_to_idx) <= self.top_k_taxonomies
        assert self.other_tax_idx == len(self.tax_to_idx)
        
        # 3. Label
        raw_label = item['labels']
        
        label_vec = np.zeros(1024, dtype=np.float32)

        # Extract all GO terms from nested structure (dict of lists)
        current_terms = []
        if isinstance(raw_label, dict):
            for val in raw_label.values():
                if isinstance(val, list): current_terms.extend(val)
                elif isinstance(val, str): current_terms.append(val)
        elif isinstance(raw_label, list):
            current_terms = raw_label
        elif isinstance(raw_label, str):
             current_terms = [raw_label]
        
        # Map to indices using loaded vocabulary
        for term in current_terms:
            if term in self.term_to_idx:
                label_vec[self.term_to_idx[term]] = 1.0
            # If term is an integer (legacy), handle it
            elif isinstance(term, int) and 0 <= term < 1024:
                 label_vec[term] = 1.0
                 
        # Fallback for dense arrays or sparse matrices
        if hasattr(raw_label, "toarray"):
            label_vec = raw_label.toarray().flatten()[:1024]
        elif isinstance(raw_label, np.ndarray):
            if raw_label.shape == (1024,):
                 label_vec = raw_label

        label = label_vec
        
        return {
            "features": torch.tensor(combined_features, dtype=torch.float32),
            "taxonomy_idx": torch.tensor(tax_idx, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32)
        }
def load_or_build_taxonomy_mapping(
    fasta_path,
    tsv_path,
    cache_path,
    top_k=1000,
    force_rebuild=False
):
    """
    Builds or loads a fixed taxonomy -> index mapping.
    """

    if os.path.exists(cache_path) and not force_rebuild:
        print(f"📦 Loading cached taxonomy mapping from {cache_path}")
        with open(cache_path, "rb") as f:
            mapping = pickle.load(f)
        return (
            mapping["tax_to_idx"],
            mapping["other_tax_idx"],
            mapping["num_taxonomies"]
        )

    print("🔨 Building taxonomy mapping from test FASTA...")

    # --- Parse FASTA ---
    protein_to_tax = {}
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split()
                if len(parts) >= 2:
                    protein_to_tax[parts[0]] = parts[1]

    # --- Parse TSV ---
    valid_tax_ids = set()
    with open(tsv_path, "r") as f:
        next(f)
        for line in f:
            tax_id = line.split("\t")[0]
            valid_tax_ids.add(tax_id)

    # --- Count frequencies ---
    counter = Counter(
        tax for tax in protein_to_tax.values()
        if tax in valid_tax_ids
    )

    top_taxonomies = [tax for tax, _ in counter.most_common(top_k)]

    tax_to_idx = {tax: i for i, tax in enumerate(top_taxonomies)}
    other_tax_idx = len(tax_to_idx)
    num_taxonomies = other_tax_idx + 1

    mapping = {
        "tax_to_idx": tax_to_idx,
        "other_tax_idx": other_tax_idx,
        "num_taxonomies": num_taxonomies,
        "top_k": top_k
    }

    # --- Save cache ---
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(mapping, f)

    print(f"💾 Saved taxonomy mapping → {cache_path}")
    print(f"✔ Taxonomies: {num_taxonomies} (incl OTHER)")
    


    return tax_to_idx, other_tax_idx, num_taxonomies

