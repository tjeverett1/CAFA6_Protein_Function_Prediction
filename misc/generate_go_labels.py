import pandas as pd
import numpy as np
from goatools.obo_parser import GODag
from tqdm import tqdm
import os


emb_dir = r"embeddings\train"
ids = [f.replace(".npz","") for f in os.listdir(emb_dir) if f.endswith(".npz")]
ids = np.array(sorted(ids))   # optional, but consistent

np.save(r"cafa-6-protein-function-prediction\Train\train_ids.npy", ids)

# ============================================================
# ======================= CONFIG ==============================
# ============================================================
CONFIG = {
    # Path to train_terms.tsv (EntryID ↦ GO term)
    "train_terms_path": r"cafa-6-protein-function-prediction\Train\train_terms.tsv",

    # Your list of proteins corresponding to embeddings
    # (this should be the same ordering as your embeddings)
    "train_ids_path": "cafa-6-protein-function-prediction/Train/train_ids.npy",

    # GO ontology
    "obo_path": r"cafa-6-protein-function-prediction/Train/go-basic.obo",

    # How many GO terms to use (top-N)
    "N_labels": 1024,

    # Output file names
    "output_labels": "labels_top1024.npy",
    "output_targets": "train_targets_top1024.npy",
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
