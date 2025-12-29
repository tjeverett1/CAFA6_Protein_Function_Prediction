import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from training_scripts.dataset import load_or_build_taxonomy_mapping, CONFIG
from training_scripts.train_mlp import ProteinFunctionMLP





import h5py
from collections import defaultdict

def load_or_build_test_metadata(
    fasta_path,
    taxon_tsv_path,
    embedding_h5_path,
    out_pickle_path,
):
    """
    Build test metadata dict:
      pid -> { "embedding": np.ndarray, "taxonomy": str }
    Only runs once; afterwards loads from pickle.
    """

    if os.path.exists(out_pickle_path):
        print(f"📦 Loading cached test metadata from {out_pickle_path}")
        with open(out_pickle_path, "rb") as f:
            return pickle.load(f)

    print("🔨 Building test metadata pickle...")

    # ------------------
    # Step 1: protein -> taxonomy from FASTA
    # ------------------
    protein_to_tax = {}

    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split()
                if len(parts) >= 2:
                    pid, tax = parts[0], parts[1]
                    protein_to_tax[pid] = tax

    print(f"✔ Parsed {len(protein_to_tax)} proteins from FASTA")

    # ------------------
    # Step 2: load valid tax IDs (optional sanity check)
    # ------------------
    valid_tax_ids = set()
    with open(taxon_tsv_path, "r") as f:
        next(f)
        for line in f:
            tax_id = line.split("\t")[0]
            valid_tax_ids.add(tax_id)

    # ------------------
    # Step 3: load embeddings from H5
    # ------------------
    metadata = {}

    with h5py.File(embedding_h5_path, "r") as h5:
        h5_ids = set(h5.keys())
        common_ids = sorted(set(protein_to_tax.keys()) & h5_ids)

        print(f"✔ Found embeddings for {len(common_ids)} / {len(protein_to_tax)} proteins")

        for pid in tqdm(common_ids, desc="Loading test embeddings"):
            emb = h5[pid][()]
            emb = emb.astype(np.float32)

            tax = protein_to_tax.get(pid)
            if tax not in valid_tax_ids:
                # still keep it, but warn once
                pass

            metadata[pid] = {
                "embedding": emb,
                "taxonomy": tax,
            }

    # ------------------
    # Step 4: save pickle
    # ------------------
    os.makedirs(os.path.dirname(out_pickle_path), exist_ok=True)
    with open(out_pickle_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"💾 Saved test metadata → {out_pickle_path}")
    print(f"✔ Metadata entries: {len(metadata)}")

    return metadata






# ---- import your model definition ----
# from your_train_script import ProteinFunctionMLP  # if in another file
# from dataset import load_or_build_taxonomy_mapping, CONFIG  # reuse your mapping cache



class ProteinEnsembleTestDataset(Dataset):
    """
    Test-time dataset: returns id, features, taxonomy_idx. No labels.
    Uses:
      - data_dict[pid]["embedding"]  (your test H5 embedding)
      - data_dict[pid]["taxonomy"]   (from FASTA)
    Optionally concatenates T5 if you actually have it; otherwise zeros.
    """
    def __init__(
        self,
        pickle_path,
        vocab_path="data/labels_top1024.npy",
        tax_to_idx=None,
        other_tax_idx=None,
        num_taxonomies=None,
        t5_pickle_path=None,  # optional
    ):
        with open(pickle_path, "rb") as f:
            self.data_dict = pickle.load(f)

        # Optional T5 dict
        self.t5_dict = {}
        if t5_pickle_path is not None and os.path.exists(t5_pickle_path):
            with open(t5_pickle_path, "rb") as f:
                self.t5_dict = pickle.load(f)

        # GO vocab (idx -> GO term)
        self.vocab = np.load(vocab_path)
        assert len(self.vocab) == 1024

        # Taxonomy mapping MUST come from checkpoint (best)
        assert tax_to_idx is not None, "Pass tax_to_idx from checkpoint['taxonomy_mapping']"
        assert other_tax_idx is not None
        assert num_taxonomies is not None

        self.tax_to_idx = tax_to_idx
        self.other_tax_idx = other_tax_idx
        self.num_taxonomies = num_taxonomies

        self.ids = list(self.data_dict.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        item = self.data_dict[pid]

        emb = item["embedding"]
        if not isinstance(emb, np.ndarray):
            emb = np.array(emb, dtype=np.float32)
        emb = emb.astype(np.float32)

        # Optional T5 concat (if you truly have it); otherwise zeros
        t5_item = self.t5_dict.get(pid)
        if t5_item is not None and isinstance(t5_item, dict) and "embedding" in t5_item:
            t5 = t5_item["embedding"]
            if not isinstance(t5, np.ndarray):
                t5 = np.array(t5, dtype=np.float32)
        else:
            t5 = np.zeros(1024, dtype=np.float32)

        feats = np.concatenate([emb, t5]).astype(np.float32)

        tax_raw = str(item.get("taxonomy"))
        tax_idx = self.tax_to_idx.get(tax_raw, self.other_tax_idx)

        return {
            "id": pid,
            "features": torch.from_numpy(feats),
            "taxonomy_idx": torch.tensor(tax_idx, dtype=torch.long),
        }

def format_score(x: float) -> str:
    """
    CAFA requirement:
      - score in (0, 1.000]
      - up to 3 significant figures
      - no zeros allowed
    """
    x = float(x)

    if x <= 0.0:
        x = 1e-6
    elif x > 1.0:
        x = 1.0

    s = f"{x:.3g}"  # 3 significant figures

    if s == "0":
        s = "1e-6"

    return s

@torch.no_grad()
def write_submission_file(
    model,
    dataset: ProteinEnsembleTestDataset,
    out_path: str,
    device: str = "cuda",
    batch_size: int = 512,
    num_workers: int = 0,
    topk_per_protein: int = 1500,
    min_prob: float = 1e-6,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    vocab = dataset.vocab  # idx -> GO term strings
    model.eval().to(device)
    lines_written = 0

    with open(out_path, "w", newline="\n") as f:
        for batch in tqdm(loader, desc="Writing submission"):
            ids = batch["id"]
            features = batch["features"].to(device)
            tax_idx = batch["taxonomy_idx"].to(device)

            logits = model(features, tax_idx)
            probs = torch.sigmoid(logits)  # (B, 1024)

            # For each protein, choose topk terms
            # (topk_per_protein can be <=1024; CAFA limit is 1500, but you only have 1024 anyway)
            k = min(topk_per_protein, probs.shape[1])
            top_vals, top_idx = torch.topk(probs, k=k, dim=1)

            top_vals = top_vals.cpu().numpy()
            top_idx = top_idx.cpu().numpy()

            for pid, vals, idxs in zip(ids, top_vals, top_idx):
                # Filter by min_prob (optional)
                for p, j in zip(vals, idxs):
                    if p < min_prob:
                        break # topk is sorted desc, so rest will be smaller
                    go_term = vocab[j]
                    # Ensure it's a plain string (np.str_ ok too)
                    go_term = str(go_term)
                    score_str = format_score(p)
                    f.write(f"{pid}\t{go_term}\t{score_str}\n")
                    lines_written += 1
    
    print("LINES WRITTEN:", lines_written)

    print(f"✅ Wrote submission file: {out_path}")

if __name__ == "__main__":
    # ---------------------------
    # Paths
    # ---------------------------
    TEST_FASTA = r"cafa-6-protein-function-prediction\Test\testsuperset.fasta"
    TEST_TAXON_TSV = r"cafa-6-protein-function-prediction\Test\testsuperset-taxon-list.tsv"
    TEST_EMB_H5 = r"data\test_embeddings.h5"

    test_pickle_path = "data/test_protein_data.pkl"
    vocab_path = "data/labels_top1024.npy"

    # ---------------------------
    # Build test metadata (if missing)
    # ---------------------------
    _ = load_or_build_test_metadata(
        fasta_path=TEST_FASTA,
        taxon_tsv_path=TEST_TAXON_TSV,
        embedding_h5_path=TEST_EMB_H5,
        out_pickle_path=test_pickle_path,
    )
    print("\n--- DEBUG TEST METADATA ---")
    with open(test_pickle_path, "rb") as f:
        d = pickle.load(f)

    print("test_pickle entries:", len(d))
    if len(d) > 0:
        k = next(iter(d))
        print("example id:", k)
        print("example emb shape:", np.array(d[k]["embedding"]).shape)
        print("example tax:", d[k].get("taxonomy"))
    else:
        print("❌ test pickle is empty!")
    print("--- END DEBUG ---\n")
    # ---------------------------
    # Load checkpoint correctly
    # ---------------------------
    ckpt_path = r"C:\Users\tessa\MIT Dropbox\Tessa Everett\6.s043\final_project\best_model_h512_lr0.001_fold0_1228_2102.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")  # dict with state + config + mapping

    state_dict = ckpt["state_dict"]
    cfg = ckpt.get("config", {})
    tax_to_idx = ckpt.get("taxonomy_mapping")  # saved mapping (topK)
    if tax_to_idx is None:
        raise ValueError("Checkpoint missing 'taxonomy_mapping'. Re-save with the new method.")

    other_tax_idx = len(tax_to_idx)
    num_taxonomies = other_tax_idx + 1

    print("✅ Checkpoint num_taxonomies:", ckpt.get("num_taxonomies", num_taxonomies))
    print("✅ Using num_taxonomies:", num_taxonomies)

    # ---------------------------
    # Build dataset using checkpoint mapping
    # ---------------------------
    test_ds = ProteinEnsembleTestDataset(
        pickle_path=test_pickle_path,
        vocab_path=vocab_path,
        tax_to_idx=tax_to_idx,
        other_tax_idx=other_tax_idx,
        num_taxonomies=num_taxonomies,
        t5_pickle_path=None,  # set this if you actually have test T5
    )

    # ---------------------------
    # Build model from checkpoint config
    # ---------------------------
    feature_dim = cfg.get("feature_dim", 2304)
    num_classes = cfg.get("num_classes", 1024)
    hidden_dim = cfg.get("hidden_dim", 512)
    tax_emb_dim = cfg.get("tax_emb_dim", 32)
    dropout = cfg.get("dropout", 0.3)

    model = ProteinFunctionMLP(
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_taxonomies=num_taxonomies,
        tax_emb_dim=tax_emb_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout,
    )

    model.load_state_dict(state_dict, strict=True)

    # ---------------------------
    # Write submission
    # ---------------------------
    out_file = "cafa_submission.tsv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    write_submission_file(
        model=model,
        dataset=test_ds,
        out_path=out_file,
        device=device,
        batch_size=512,
        num_workers=0,
        topk_per_protein=1500,
        min_prob=1e-6
    )
