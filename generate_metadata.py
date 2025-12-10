import os
import glob
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import h5py
import pickle
import numpy as np

# ------------------------
# CONFIG
# ------------------------
T5_H5_PATH = r"embeddings\per-protein.h5"
FOLDDIR    = "splits"
TAX_PATH   = r"cafa-6-protein-function-prediction\Train\train_taxonomy.tsv"
TERMS_PATH = r"cafa-6-protein-function-prediction\Train\train_terms.tsv"
OUTFILE    = "t5_metadata.pkl"

# ------------------------
# STEP 1: Collect all relevant protein IDs
# ------------------------

needed_ids = set()

# Load folds
fold_files = sorted(glob.glob(os.path.join(FOLDDIR, "fold_*.tsv")))
print("Collecting protein IDs from fold files...")

id_to_fold = {}
for fpath in tqdm(fold_files, desc="Folds"):
    fold_id = int(os.path.basename(fpath).split("_")[1].split(".")[0])
    df = pd.read_csv(fpath, sep="\t")
    for pid in df["protein"]:
        needed_ids.add(pid)
        id_to_fold[pid] = fold_id

print(f"IDs from folds: {len(id_to_fold)}")

# Load taxonomy
print("Collecting protein IDs from taxonomy...")
tax = pd.read_csv(TAX_PATH, sep="\t", header=None, names=["EntryID", "taxonomy"])
id_to_tax = dict(zip(tax.EntryID, tax.taxonomy))

needed_ids.update(id_to_tax.keys())
print(f"IDs after adding taxonomy: {len(needed_ids)}")

# Load GO terms
print("Collecting protein IDs from GO labels...")
terms = pd.read_csv(TERMS_PATH, sep="\t")
labels = defaultdict(lambda: {"MF": [], "BP": [], "CC": []})

for _, row in tqdm(terms.iterrows(), total=len(terms), desc="GO terms"):
    pid, go, asp = row["EntryID"], row["term"], row["aspect"]
    needed_ids.add(pid)
    if asp == "F": labels[pid]["MF"].append(go)
    elif asp == "P": labels[pid]["BP"].append(go)
    elif asp == "C": labels[pid]["CC"].append(go)

print(f"FINAL number of proteins needed: {len(needed_ids)}\n")


# ------------------------
# STEP 2: Load ONLY needed embeddings
# ------------------------

print("Loading required embeddings only...")

id_to_embedding = {}
with h5py.File(T5_H5_PATH, "r") as f:

    # intersect needed IDs with H5 keys
    h5_ids = set(f.keys())
    to_load = sorted(needed_ids & h5_ids)

    print(f"Embeddings found for {len(to_load)} of {len(needed_ids)} proteins")

    for pid in tqdm(to_load, desc="Embedding load"):
        vec = f[pid][()]               # dataset -> numpy array
        vec = vec.astype(np.float32)   # convert float16 → float32
        id_to_embedding[pid] = vec

print(f"Loaded {len(id_to_embedding)} embeddings\n")

# ------------------------
# STEP 3: Build metadata
# ------------------------

print("Building metadata dictionary...")

metadata = {}

for pid in tqdm(id_to_embedding.keys(), desc="Metadata assembly"):
    metadata[pid] = {
        "embedding": id_to_embedding[pid],
        "taxonomy": id_to_tax.get(pid),
        "fold": id_to_fold.get(pid),
        "labels": labels.get(pid, {"MF": [], "BP": [], "CC": []}),
    }

print(f"Metadata created for {len(metadata)} proteins")

# ------------------------
# STEP 4: Save
# ------------------------

print(f"Saving metadata to {OUTFILE}")
with open(OUTFILE, "wb") as f:
    pickle.dump(metadata, f)
print("Saved!\n")


# ------------------------
# STEP 5: Print first 3 entries
# ------------------------
print("Example entries:\n")
for pid in list(metadata.keys())[:3]:
    print("Protein:", pid)
    print("Embedding shape:", metadata[pid]["embedding"].shape)
    print("Taxonomy:", metadata[pid]["taxonomy"])
    print("Fold:", metadata[pid]["fold"])
    print("Labels:", metadata[pid]["labels"])
    print("---")
