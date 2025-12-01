import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class ProteinEnsembleDataset(Dataset):
    def __init__(self, pickle_path, mode='train', val_fold=0):
        """
        Args:
            pickle_path (str): Path to the massive dictionary pickle.
            mode (str): 'train' or 'val'.
            val_fold (int): The fold ID to use for validation.
                            If mode='train', excludes this fold.
                            If mode='val', includes ONLY this fold.
        """
        print(f"📦 Loading dataset from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            self.data_dict = pickle.load(f)
            
        # Get list of all IDs
        all_ids = list(self.data_dict.keys())
        
        # 1. Build Taxonomy Map (Fixed across all folds)
        # We need a consistent mapping from TaxID -> Index (0..N)
        self.tax_ids = sorted(list(set(d['taxonomy'] for d in self.data_dict.values())))
        self.tax_to_idx = {tax: i for i, tax in enumerate(self.tax_ids)}
        self.num_taxonomies = len(self.tax_ids)
        print(f"✔ Found {self.num_taxonomies} unique taxonomies.")

        # 2. Filter by Fold
        self.filtered_ids = []
        
        for pid in all_ids:
            item = self.data_dict[pid]
            item_fold = item.get('fold') # Default to -1 if missing? Assuming exists.
            
            if mode == 'train':
                if item_fold != val_fold:
                    self.filtered_ids.append(pid)
            elif mode == 'val':
                if item_fold == val_fold:
                    self.filtered_ids.append(pid)
        
        print(f"✔ {mode.upper()} set: {len(self.filtered_ids)} samples (Val Fold: {val_fold})")

    def __len__(self):
        return len(self.filtered_ids)

    def __getitem__(self, idx):
        pid = self.filtered_ids[idx]
        item = self.data_dict[pid]
        
        # 1. Features
        # Concatenate ESM (1280) + T5 (1024)
        esm = item['esm_embedding']
        t5 = item['t5_embedding']
        
        # Ensure they are numpy arrays (just in case)
        if not isinstance(esm, np.ndarray): esm = np.array(esm)
        if not isinstance(t5, np.ndarray): t5 = np.array(t5)
            
        combined_features = np.concatenate([esm, t5])
        
        # 2. Taxonomy Index
        tax_raw = item['taxonomy']
        tax_idx = self.tax_to_idx.get(tax_raw, 0) # Default 0 if unseen
        
        # 3. Label
        label = item['label']
        
        return {
            "features": torch.tensor(combined_features, dtype=torch.float32),
            "taxonomy_idx": torch.tensor(tax_idx, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32)
        }

