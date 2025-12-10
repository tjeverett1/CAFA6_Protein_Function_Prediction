import pandas as pd

# Load file
df = pd.read_csv(r"cafa-6-protein-function-prediction\Train\train_terms.tsv", sep="\t")

# Count frequencies of each GO term
term_counts = df["term"].value_counts()

# Select top 1024 most common GO terms
top_1024_terms = term_counts.index[:1024]

print("Top 10 GO terms:")
print(top_1024_terms[:10])

# Extract aspects for each GO term
# Because each GO term can appear multiple times, take the majority aspect or first occurrence
go_to_aspect = df.drop_duplicates(subset=["term"]).set_index("term")["aspect"].to_dict()

# Count aspects among top-1024
aspect_counts = {"F": 0, "P": 0, "C": 0}
for go in top_1024_terms:
    asp = go_to_aspect.get(go, None)
    if asp in aspect_counts:
        aspect_counts[asp] += 1

# Compute percentages
total = len(top_1024_terms)
aspect_percent = {k: (v / total) * 100 for k, v in aspect_counts.items()}

print("\nAspect breakdown among top 1024 GO terms:")
print(f"Molecular Function (F): {aspect_counts['F']} terms ({aspect_percent['F']:.2f}%)")
print(f"Biological Process (P): {aspect_counts['P']} terms ({aspect_percent['P']:.2f}%)")
print(f"Cellular Component (C): {aspect_counts['C']} terms ({aspect_percent['C']:.2f}%)")
