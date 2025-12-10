import pandas as pd
import numpy as np
from goatools.obo_parser import GODag
from tqdm import tqdm
import os


emb_dir = "data/train_embeddings"

def extract_accession(filename):
    # Input: sp_A0A023FBW4_E1142_AMBCJ.npz
    name = filename.replace(".npz", "")
    parts = name.split("_")
    # Robust extraction: typically index 1
    if len(parts) >= 2:
        return parts[1]
    return name 

# Always regenerate to ensure consistency
files = [f for f in os.listdir(emb_dir) if f.endswith(".npz")]
files.sort() # Deterministic order

# 1. Save filenames (for loading X)
np.save("data/train_filenames.npy", np.array(files))

# 2. Save clean IDs (for matching Y)
clean_ids = [extract_accession(f) for f in files]
np.save("data/train_ids.npy", np.array(clean_ids))

print(f"✅ Regenerated IDs. Found {len(files)} embeddings.")
print(f"Sample Filename: {files[0]}")
print(f"Sample ID: {clean_ids[0]}")


# ============================================================
# ======================= CONFIG ==============================
# ============================================================
CONFIG = {
    "train_terms_path": "data/train_terms.tsv",
    "train_ids_path": "data/train_ids.npy", # Use clean IDs for label matching
    "obo_path": "data/go-basic.obo",
    "N_labels": 1024,
    "output_labels": "data/labels_top1024.npy",
    "output_targets": "data/train_targets_top1024.npy",
}
# ============================================================


# ------------------------------------------------------------
# Load GO DAG (ontology graph)
# ------------------------------------------------------------
def load_go_dag(obo_path):
    print("📦 Loading GO ontology...")
    go = GODag(obo_path)
    print(f"✔ Loaded {len(go)} GO terms from ontology")
    return go


# ------------------------------------------------------------
# Expand GO terms upward to ancestors 
# (CAFA standard propagation)
# ------------------------------------------------------------
def get_ancestors(go_id, go_dag):
    """Return all ancestors including itself."""
    if go_id not in go_dag:
        return []
    return [go_id] + list(go_dag[go_id].get_all_parents())


# ------------------------------------------------------------
# Build label vectors
# ------------------------------------------------------------
def build_label_matrix(train_ids, terms_df, top_terms, go_dag):
    term_to_idx = {term: i for i, term in enumerate(top_terms)}
    N = len(top_terms)

    # Allocate matrix
    Y = np.zeros((len(train_ids), N), dtype=np.int8)

    # Group GO annotations by protein
    grouped = terms_df.groupby("EntryID")["term"].apply(list)

    print("🧬 Generating CAFA-style label matrix...")
    for i, pid in enumerate(tqdm(train_ids)):
        if pid not in grouped:
            continue
        
        raw_terms = grouped[pid]

        # Ancestor propagation
        expanded = []
        for t in raw_terms:
            expanded += get_ancestors(t, go_dag)

        for t in expanded:
            if t in term_to_idx:
                Y[i, term_to_idx[t]] = 1

    return Y


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    print("📄 Loading input files...")
    terms_df = pd.read_csv(CONFIG["train_terms_path"], sep="\t")

    train_ids = list(np.load(CONFIG["train_ids_path"]))

    go_dag = load_go_dag(CONFIG["obo_path"])

    # Pick most frequent GO terms
    N = CONFIG["N_labels"]
    print(f"📊 Selecting top {N} GO terms...")
    top_terms = (
        terms_df["term"]
        .value_counts()
        .index[:N]
        .tolist()
    )
    print(f"✔ Selected {len(top_terms)} frequent GO terms")

    # Build label matrix
    Y = build_label_matrix(train_ids, terms_df, top_terms, go_dag)

    # Save
    print("💾 Saving output files...")
    np.save(CONFIG["output_targets"], Y)
    np.save(CONFIG["output_labels"], np.array(top_terms))

    print("\n🎉 DONE!")
    print(f"Saved: {CONFIG['output_targets']} (shape={Y.shape})")
    print(f"Saved: {CONFIG['output_labels']} (GO vocabulary)")


if __name__ == "__main__":
    main()
