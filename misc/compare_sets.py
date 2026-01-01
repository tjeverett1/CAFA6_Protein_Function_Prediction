import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load embeddings
train_embeds = np.load("train_embeds.npy")   # (N_train, D)
test_embeds  = np.load("test_embeds.npy")    # (N_test, D)

# Normalize rows (makes cosine similarity much faster)
def normalize(mat):
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)

train_norm = normalize(train_embeds)
test_norm  = normalize(test_embeds)

batch_size = 2000   # adjust to fit your GPU/CPU RAM

nearest_sim = []

print("Computing nearest-neighbor similarity…")
for i in tqdm(range(0, len(test_norm), batch_size)):
    batch = test_norm[i:i+batch_size]
    
    # cosine similarities: (batch_size × N_train)
    sims = batch @ train_norm.T
    
    # best similarity to ANY train protein
    best = sims.max(axis=1)
    nearest_sim.extend(best.tolist())

nearest_sim = np.array(nearest_sim)

print("Mean similarity:", nearest_sim.mean())
print("Median similarity:", np.median(nearest_sim))

# bins
for thr in [0.9, 0.8, 0.7, 0.6, 0.5, 0.3]:
    frac = np.mean(nearest_sim >= thr)
    print(f"Fraction >= {thr}: {frac:.3f}")

# histogram if needed
import matplotlib.pyplot as plt
plt.hist(nearest_sim, bins=50)
plt.xlabel("Max cosine similarity to train set")
plt.ylabel("Count")
plt.show()
