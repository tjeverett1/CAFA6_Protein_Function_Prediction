import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from esm import pretrained

# ============================================================
# CONFIGURATION
# ============================================================

FASTA_PATH = r"testsuperset.fasta"          # your input FASTA
OUT_DIR = r"/orcd/home/002/tjeveret/orcd/pool/embeddings/test"                 # where per-protein .npz embeddings go
MODEL_NAME = "esm2_t33_650M_UR50D"       # good quality + fits on 4GB GPU

BATCH_BASE = 4                           # base batch size for ~300–600 aa
MAX_NORMAL_LEN = 5000                    # sequences <= this length processed normally
CHUNK_SIZE = 1024                        # chunk size for long proteins
CHUNK_OVERLAP = 128     
TRAIN_EMB_DIR = "/orcd/home/002/tjeveret/orcd/pool/embeddings/train"                 # optional overlap for smoother chunking

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sanitize_id(raw_id):
    """Clean FASTA IDs to safe filenames."""
    return raw_id.replace("|", "_").replace("/", "_").replace(" ", "_")


def chunk_sequence(seq, chunk_size=1024, overlap=0):
    """Split huge proteins into overlapping chunks."""
    chunks = []
    L = len(seq)
    step = chunk_size - overlap
    for i in range(0, L, step):
        chunks.append(seq[i:i+chunk_size])
    return chunks


def embed_chunked(seq, model, batch_converter, device):
    """Embed very long sequences using chunking + averaging."""
    seq_chunks = chunk_sequence(seq, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_vecs = []

    for chunk in seq_chunks:
        data = [("chunk", str(chunk))]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)

        with torch.no_grad():
            out = model(toks, repr_layers=[model.num_layers])
        rep = out["representations"][model.num_layers][0]  # (L, D)
        rep = rep[1:-1].cpu().numpy()                     # remove CLS/EOS
        chunk_vecs.append(rep.mean(axis=0))

    # Average across chunks → final embedding
    protein_vec = np.mean(chunk_vecs, axis=0)
    return protein_vec


def save_embedding(pid, prot_emb):
    out_path = os.path.join(OUT_DIR, f"{pid}.npz")
    np.savez_compressed(out_path, embedding=prot_emb)


def already_done(pid):
    return os.path.exists(os.path.join(OUT_DIR, f"{pid}.npz"))

# ============================================================
# PRELOAD EXISTING EMBEDDINGS
# ============================================================

def load_existing_ids():
    existing = set()

    # test embeddings
    for f in os.listdir(OUT_DIR):
        if f.endswith(".npz"):
            existing.add(f[:-4])

    # train embeddings
    for f in os.listdir(TRAIN_EMB_DIR):
        if f.endswith(".npz"):
            existing.add(f[:-4])

    print(f"✔ Found {len(existing)} embeddings already computed.")
    return existing



# ============================================================
# MAIN EMBEDDING FUNCTION
# ============================================================

def embed_fasta_chunked():

    # ---------- GPU INFO ----------
    print(f"🔥 Using device: {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"🧠 GPU: {props.name} | VRAM: {props.total_memory/1024**3:.2f} GB")

    # ---------- Load ESM2 model ----------
    print(f"\n📦 Loading model: {MODEL_NAME}")
    model_loader = getattr(pretrained, MODEL_NAME)
    model, alphabet = model_loader()
    model = model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    print("✔️ Model loaded.\n")

    # ---------- Read FASTA ----------
    records = list(SeqIO.parse(FASTA_PATH, "fasta"))
    print(f"📄 Loaded {len(records)} sequences.")

    # ---------- Sort by length ----------
    lengths = [len(r.seq) for r in records]
    print(f"📏 Max length in FASTA: {max(lengths)}")
    records.sort(key=lambda r: len(r.seq))
    print("✔️ Sequences sorted by length.\n")

    existing_ids = load_existing_ids()

    # ---------- Process each record ----------
    for rec in tqdm(records, desc="Embedding proteins"):

        pid = sanitize_id(rec.id)
        seq = str(rec.seq)
        L = len(seq)

        # Resume mode
        if pid in existing_ids:
            continue

        # --------------------------
        # CASE 1: Short/Normal sequences
        # --------------------------
        if L <= MAX_NORMAL_LEN:
            # dynamic batch = smaller for long sequences
            batch_size = max(1, BATCH_BASE // max(1, L // 800))

            data = [(pid, seq)]
            _, _, toks = batch_converter(data)
            toks = toks.to(DEVICE)

            try:
                with torch.no_grad():
                    out = model(toks, repr_layers=[model.num_layers])
                rep = out["representations"][model.num_layers][0]
                rep = rep[1:-1].cpu().numpy()
                prot_vec = rep.mean(axis=0)

            except torch.cuda.OutOfMemoryError:
                print(f"\n⚠️ OOM for {pid} ({L} aa) → switching to chunked mode.")
                prot_vec = embed_chunked(seq, model, batch_converter, DEVICE)

        # --------------------------
        # CASE 2: Very long sequences
        # --------------------------
        else:
            print(f"\n⚠️ {pid} is long ({L} aa) → using chunking.")
            prot_vec = embed_chunked(seq, model, batch_converter, DEVICE)

        # --------------------------
        # Save result
        # --------------------------
        save_embedding(pid, prot_vec)

    print("\n🎉 Done! All embeddings saved.")


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    embed_fasta_chunked()
    
    