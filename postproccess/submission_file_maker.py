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
from postproccess.goa_boost import build_goa_lookup_for_proteins, apply_goa_boost_inplace,  load_or_build_goa_lookup, load_or_build_goa_pos_neg_lookup, apply_goa_pos_neg_constraints_inplace



LINEAGE_PKL = r"C:\Users\tessa\MIT Dropbox\Tessa Everett\6.s043\final_project\data\taxonomy_lineage_mapping.pkl"

with open(LINEAGE_PKL, "rb") as f:
    lineage_obj = pickle.load(f)

node_to_idx = lineage_obj["node_to_idx"]          # dict: taxid(str)->int, PAD "0"->0
lineage_table = lineage_obj["lineage_table"]      # dict: taxid(str)->list[str] length R
num_ranks = int(lineage_obj["num_ranks"])
num_tax_nodes = int(lineage_obj["num_nodes"])     # == len(node_to_idx)

print("✅ Loaded lineage mapping:",
      "num_tax_nodes=", num_tax_nodes,
      "num_ranks=", num_ranks)



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
    Test dataset that can return either:
      - taxonomy_idx (old)  OR
      - lineage_idx  (new)
    depending on tax_repr.
    """
    def __init__(
        self,
        pickle_path,
        vocab_path="data/labels_top1024.npy",
        t5_pickle_path=None,

        # new repr:
        lineage_table=None,   # dict tax_id(str) -> list[str] length R (taxids)
        node_to_idx=None,     # dict taxid(str)->int, PAD="0"->0
        num_ranks=None,

        tax_repr="lineage",  # "taxonomy" or "lineage"
    ):
        with open(pickle_path, "rb") as f:
            self.data_dict = pickle.load(f)

        self.t5_dict = {}
        if t5_pickle_path is not None and os.path.exists(t5_pickle_path):
            with open(t5_pickle_path, "rb") as f:
                self.t5_dict = pickle.load(f)

        self.vocab = np.load(vocab_path)
        assert len(self.vocab) == 1024

        self.tax_repr = tax_repr


        
        assert lineage_table is not None and node_to_idx is not None and num_ranks is not None
        self.lineage_table = lineage_table
        self.node_to_idx = node_to_idx
        self.num_ranks = int(num_ranks)


        self.ids = list(self.data_dict.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        item = self.data_dict[pid]

        emb = np.asarray(item["embedding"], dtype=np.float32)

        t5_item = self.t5_dict.get(pid)
        if t5_item is not None and isinstance(t5_item, dict) and "embedding" in t5_item:
            t5 = np.asarray(t5_item["embedding"], dtype=np.float32)
        else:
            t5 = np.zeros(1024, dtype=np.float32)

        # feats = np.concatenate([emb, t5]).astype(np.float32)
        feats = t5.astype(np.float32)

        tax_raw = str(item.get("taxonomy"))

        out = {
            "id": pid,
            "features": torch.from_numpy(feats),
        }

        if self.tax_repr == "taxonomy":
            tax_idx = self.tax_to_idx.get(tax_raw, self.other_tax_idx)
            out["taxonomy_idx"] = torch.tensor(tax_idx, dtype=torch.long)

        else:  # lineage
            lineage_nodes = self.lineage_table.get(tax_raw, [tax_raw] + ["0"] * (self.num_ranks - 1))
            lineage_idx = [self.node_to_idx.get(n, 0) for n in lineage_nodes]
            out["lineage_idx"] = torch.tensor(lineage_idx, dtype=torch.long)

        return out

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
def write_ensemble_submission_file_lineage(
    ckpt_paths,
    dataset,
    out_path,
    device="cuda",
    batch_size=512,
    num_workers=0,
    topk_per_protein=1500,
    min_prob=1e-6,
    goa_lookup=None,         # NEW
    goa_boost=0.5,           # NEW
    goa_mode="max",          # NEW
    goa_pos= None,
    goa_neg=None
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    vocab = dataset.vocab

    # ---- Load lineage models ----
    models = []
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})

        feature_dim = cfg.get("feature_dim", 2304)
        num_classes = cfg.get("num_classes", 1024)
        hidden_dim  = cfg.get("hidden_dim", 512)
        tax_emb_dim = cfg.get("tax_emb_dim", 32)
        dropout     = cfg.get("dropout", 0.3)

        # Lineage-only model ctor
        m = ProteinFunctionMLP(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_tax_nodes=num_tax_nodes,
            num_ranks=num_ranks,
            tax_emb_dim=tax_emb_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
        )
        m.load_state_dict(ckpt["state_dict"], strict=True)
        m.to(device).eval()
        models.append(m)

    # ---- Write submission ----
    lines_written = 0
    with open(out_path, "w", newline="\n") as f:
        for batch in tqdm(loader, desc="Writing ensemble submission (lineage)"):
            ids = batch["id"]
            features = batch["features"].to(device)
            lineage_idx = batch["lineage_idx"].to(device)  # [B, R]

            probs_sum = None
            for m in models:
                logits = m(features, lineage_idx)
                probs = torch.sigmoid(logits)
                probs_sum = probs if probs_sum is None else (probs_sum + probs)

            probs = probs_sum / len(models)  # [B, 1024]

            if goa_pos is not None:
                probs_np = probs.detach().cpu().numpy()

                apply_goa_pos_neg_constraints_inplace(
                    ids=ids,
                    probs=probs_np,
                    vocab=vocab,
                    goa_pos=goa_pos,
                    goa_neg=goa_neg,
                    descendants_map=None,          # set later if you have GO DAG
                    neg_cap=1e-4,
                    descendant_cap=1e-4,
                    descendant_conf_threshold=0.2,
                )

                probs = torch.from_numpy(probs_np).to(device)
            elif goa_lookup is not None:
                probs_np = probs.detach().cpu().numpy()
                apply_goa_boost_inplace(
                    ids=ids,
                    probs=probs_np,
                    vocab=vocab,
                    goa_lookup=goa_lookup,
                    boost=goa_boost,
                    mode=goa_mode,
                )
                probs = torch.from_numpy(probs_np).to(device)

            k = min(topk_per_protein, probs.shape[1])
            top_vals, top_idx = torch.topk(probs, k=k, dim=1)

            top_vals = top_vals.cpu().numpy()
            top_idx = top_idx.cpu().numpy()

            for pid, vals, idxs in zip(ids, top_vals, top_idx):
                for pval, j in zip(vals, idxs):
                    if pval < min_prob:
                        break
                    go_term = str(vocab[j])
                    f.write(f"{pid}\t{go_term}\t{format_score(pval)}\n")
                    lines_written += 1

    print("LINES WRITTEN:", lines_written)
    print(f"✅ Wrote submission file: {out_path}")



if __name__ == "__main__":
    # ============================
    # CONFIG: paths
    # ============================
    TEST_FASTA = r"cafa-6-protein-function-prediction\Test\testsuperset.fasta"
    TEST_TAXON_TSV = r"cafa-6-protein-function-prediction\Test\testsuperset-taxon-list.tsv"
    TEST_EMB_H5 = r"data\test_embeddings.h5"
    TEST_PICKLE = "data/test_protein_data.pkl"
    VOCAB_PATH = "data/labels_top1024.npy"

    # Lineage mapping (for lineage-based models)
    LINEAGE_PKL = r"C:\Users\tessa\MIT Dropbox\Tessa Everett\6.s043\final_project\data\taxonomy_lineage_mapping.pkl"

    # ============================
    # CHECKPOINTS (list length 1 or >1)
    # ============================
    MODEL_DIR = r"C:\Users\tessa\MIT Dropbox\Tessa Everett\6.s043\final_project\models\kfold5"
    CKPT_PATHS = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]

    OUT_FILE = "submissions/2025_12_31/sub2/submission.tsv"
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Runtime
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 512
    NUM_WORKERS = 0
    TOPK_PER_PROTEIN = 1024
    MIN_PROB = 1e-6

    # ============================
    # 1) Build/load test metadata pickle
    # ============================
    _ = load_or_build_test_metadata(
        fasta_path=TEST_FASTA,
        taxon_tsv_path=TEST_TAXON_TSV,
        embedding_h5_path=TEST_EMB_H5,
        out_pickle_path=TEST_PICKLE,
    )

    # ============================
    # 2) Build test dataset
    # ============================
    with open(LINEAGE_PKL, "rb") as f:
        lineage_obj = pickle.load(f)

    node_to_idx = lineage_obj["node_to_idx"]
    lineage_table = lineage_obj["lineage_table"]
    num_ranks = int(lineage_obj["num_ranks"])
    num_tax_nodes = int(lineage_obj["num_nodes"])

    test_ds = ProteinEnsembleTestDataset(
        pickle_path=TEST_PICKLE,
        vocab_path=VOCAB_PATH,
        lineage_table=lineage_table,
        node_to_idx=node_to_idx,
        num_ranks=num_ranks,
        tax_repr="lineage",
        t5_pickle_path=None,
    )

    USE_GOA_BOOST = True

    GOA_GAF_PATH = r"C:\Users\tessa\Desktop\CAFA\go_uniprot_all\goa_uniprot_all.gaf\goa_uniprot_all.gaf"
    GOA_CACHE_PKL = r"C:\Users\tessa\Desktop\CAFA\cache\goa_lookup_test.pkl"

    GOA_ASPECTS = None      # or {"F","P","C"} if you want to restrict
    GOA_BOOST_VALUE = .5
    GOA_BOOST_MODE = "max"

    GOA_POSNEG_CACHE = r"C:\Users\tessa\Desktop\CAFA\cache\goa_posneg_test.pkl"

    goa_pos, goa_neg = None, None
    if USE_GOA_BOOST:
        # goa_pos, goa_neg
        goa_lookup = load_or_build_goa_lookup(
            gaf_path=GOA_GAF_PATH,
            proteins=test_ds.ids,
            cache_path= GOA_CACHE_PKL ,
            allowed_aspect=GOA_ASPECTS,
        )
    # ============================
    # 4) Always run ensemble inference
    # ============================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    write_ensemble_submission_file_lineage(
    ckpt_paths=CKPT_PATHS,
    dataset=test_ds,
    out_path=OUT_FILE,
    device=device,
    batch_size=512,
    num_workers=0,
    topk_per_protein=1500,
    min_prob=1e-6,
    goa_lookup= goa_lookup,
    goa_boost=GOA_BOOST_VALUE,
    goa_mode=GOA_BOOST_MODE,
    goa_pos=None, #goa_pos,
    goa_neg= None, #goa_neg,
)

