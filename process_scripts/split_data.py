import pandas as pd
from Bio import SeqIO
from collections import defaultdict
import numpy as np
import random
import os

# ---------------------------
# CONFIG
# ---------------------------
CLUSTER_TSV = r"mmseq_out\train_linclust_cluster.tsv"            # your MMseqs output
FASTA_PATH   = r"cafa-6-protein-function-prediction\Train\train_sequences.fasta"        # your protein FASTA
TERMS_PATH   = r"cafa-6-protein-function-prediction\Train\train_terms.tsv"              # EntryID, term, aspect
OUTDIR       = "splits2"                       # output directory
TEST_FRAC    = 0.20                           # percent of clusters in test set
K            = 5                              # K-fold CV
random.seed(42)

os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------
# Load cluster TSV
# ---------------------------
df = pd.read_csv(CLUSTER_TSV, sep="\t", header=None,
                 names=["cluster", "protein"])

clusters = defaultdict(list)
for _, row in df.iterrows():
    clusters[row["cluster"]].append(row["protein"])

cluster_list = list(clusters.values())
print(f'cluster_list: {cluster_list[:10]}')
print(f"Loaded {len(cluster_list)} clusters.")

# ---------------------------
# Homology-aware train/test split
# ---------------------------
random.shuffle(cluster_list)
n_test = int(len(cluster_list) * TEST_FRAC)

test_clusters = cluster_list[:n_test]
train_clusters = cluster_list[n_test:]

train_proteins = {p for c in train_clusters for p in c}
test_proteins  = {p for c in test_clusters  for p in c}

print(f"Train proteins: {len(train_proteins)}")
print(f"Test proteins:  {len(test_proteins)}")

# ---------------------------
# Load FASTA records
# ---------------------------
records = SeqIO.to_dict(SeqIO.parse(FASTA_PATH, "fasta"))

SeqIO.write(
    [records[p] for p in train_proteins if p in records],
    f"{OUTDIR}/train_split.fasta", "fasta"
)
SeqIO.write(
    [records[p] for p in test_proteins if p in records],
    f"{OUTDIR}/test_split.fasta", "fasta"
)

# ---------------------------
# Split GO terms
# ---------------------------
terms = pd.read_csv(TERMS_PATH, sep="\t")

terms_train = terms[terms.EntryID.isin(train_proteins)]
terms_test  = terms[terms.EntryID.isin(test_proteins)]

terms_train.to_csv(f"{OUTDIR}/train_terms.tsv", sep="\t", index=False)
terms_test.to_csv(f"{OUTDIR}/test_terms.tsv", sep="\t", index=False)

# ---------------------------
# Create K-fold splits (cluster-based)
# ---------------------------
folds = [[] for _ in range(K)]
sizes = [0] * K

# Greedy balancing of fold sizes
for cluster in cluster_list:
    i = np.argmin(sizes)
    folds[i].extend(cluster)
    sizes[i] += len(cluster)

for i in range(K):
    fold_df = pd.DataFrame({"protein": folds[i]})
    fold_df.to_csv(f"{OUTDIR}/fold_{i}.tsv", sep="\t", index=False)

print("\nAll splits written to:", OUTDIR)
