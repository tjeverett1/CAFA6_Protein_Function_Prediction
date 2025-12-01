import os
import torch
import h5py
import numpy as np
from Bio import SeqIO
from esm import pretrained
from tqdm import tqdm

# --------------------------------------------------------------
# Utility: save embeddings to HDF5
# --------------------------------------------------------------
def save_h5(out_path, seq, residue_emb, seq_emb):
    with h5py.File(out_path, "w") as f:
        f.create_dataset("sequence", data=np.string_(seq))
        f.create_dataset("residue_emb", data=residue_emb.astype(np.float32))
        f.create_dataset("seq_emb", data=seq_emb.astype(np.float32))


# --------------------------------------------------------------
# Main embedding function
# --------------------------------------------------------------
def embed_fasta_batched(
    fasta_path,
    out_dir,
    model_name="esm2_t33_650M_UR50D",
    batch_size=8,
):
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------------
    # Determine device
    # --------------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🔥 Using GPU: {gpu_name}")
        print(f"🧠 GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("⚠️ CUDA not available! Using CPU — this will be slow.")

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------
    print(f"\n📦 Loading model: {model_name}")
    model_loader = getattr(pretrained, model_name)
    model, alphabet = model_loader()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    print("✔️ Model loaded.")

    # --------------------------------------------------------------
    # Load FASTA sequences
    # --------------------------------------------------------------
    records = list(SeqIO.parse(fasta_path, "fasta"))
    print(f"\n📄 Loaded {len(records)} sequences from FASTA.")
    longest = max(len(r.seq) for r in records)
    print(f"📏 Longest sequence length: {longest}")

    # Sort sequences by length to reduce padding
    records.sort(key=lambda x: len(x.seq), reverse=True)
    print("✔️ Sequences sorted by length.")

    # --------------------------------------------------------------
    # Batch embedding
    # --------------------------------------------------------------
    print("\n🚀 Starting embedding...")
    pbar = tqdm(range(0, len(records), batch_size))

    for idx in pbar:
        batch_records = records[idx : idx + batch_size]
        batch_data = [(rec.id, str(rec.seq)) for rec in batch_records]

        # Show progress info
        longest_in_batch = max(len(r.seq) for r in batch_records)
        pbar.set_description(f"Batch {idx//batch_size+1} | max_len={longest_in_batch}")

        # Convert to tokens
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, repr_layers=[model.num_layers])

        reps = out["representations"][model.num_layers]

        # Save each protein
        for rec, rep in zip(batch_records, reps):
            # Extract UniProt accession cleanly
            raw_id = rec.id
            if "|" in raw_id:
                pid = raw_id.split("|")[1]     # clean ID e.g., A0A0C5B5G6
            else:
                pid = raw_id                   # fallback

            seq = str(rec.seq)

            rep = rep.cpu().numpy()
            residue_emb = rep[1:-1]
            seq_emb = residue_emb.mean(axis=0)

            out_path = os.path.join(out_dir, f"{pid}.h5")
            save_h5(out_path, seq, residue_emb, seq_emb)

    print("\n🎉 Done! All embeddings saved to:", out_dir)

if __name__ == "__main__":
    
    embed_fasta_batched(
        fasta_path=r"C:\Users\tessa\MIT Dropbox\Tessa Everett\6.s043\final_project\cafa-6-protein-function-prediction\Train\train_sequences.fasta",
        out_dir=r"C:\Users\tessa\MIT Dropbox\Tessa Everett\6.s043\final_project\embeddings\train",
        model_name="esm2_t33_650M_UR50D",  # or esm2_t36_3B_UR50D
        batch_size=8,
    
    )