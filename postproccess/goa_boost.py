# post_processing/goa_boost.py
import gzip
from typing import Dict, Set, Iterable, Tuple, Optional
import numpy as np
import os
import pickle



def _open_text_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def build_goa_lookup_for_proteins(
    gaf_path: str,
    proteins: Iterable[str],
    allowed_aspect: Optional[Set[str]] = None,  # {"F","P","C"} or None
) -> Dict[str, Set[str]]:
    """
    Build mapping: protein_id -> set(GO terms) using GOA UniProt GAF.
    - Skips comment lines starting with "!"
    - Drops any row where Qualifier contains "NOT"
    - Only keeps rows where DB_Object_ID is in `proteins`
    - Optionally filter by Aspect (col 9): F/P/C
    GAF 2.2 columns (1-indexed):
      2 DB_Object_ID
      4 Qualifier
      5 GO_ID
      9 Aspect
    """
    proteins = set(proteins)
    out: Dict[str, Set[str]] = {}

    with _open_text_maybe_gz(gaf_path) as f:
        for line in f:
            if not line or line[0] == "!":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            prot = parts[1]        # DB_Object_ID
            if prot not in proteins:
                continue

            qualifier = parts[3]   # Qualifier
            if "NOT" in qualifier:
                continue

            go_id = parts[4]       # GO_ID
            aspect = parts[8]      # Aspect: F/P/C

            if allowed_aspect is not None and aspect not in allowed_aspect:
                continue

            out.setdefault(prot, set()).add(go_id)

    return out


def apply_goa_boost_inplace(
    ids: Iterable[str],
    probs: np.ndarray,
    vocab: np.ndarray,
    goa_lookup: Dict[str, Set[str]],
    boost: float = 0.5,
    mode: str = "max",
) -> Tuple[int, int]:
    """
    Apply GOA boost to probability matrix.
    Args:
      ids: iterable of protein ids length B
      probs: np array shape (B, num_terms) in [0,1]
      vocab: np array shape (num_terms,), idx -> GO term string
      goa_lookup: dict protein -> set(GO strings)
      boost: probability to set/raise to
      mode: "max" (p = max(p, boost)) or "set" (p = boost)
    Returns:
      (num_proteins_touched, num_term_updates)
    """
    # Build GO term -> index map once (fast lookup)
    term_to_idx = {str(t): i for i, t in enumerate(vocab)}

    touched = 0
    updates = 0

    for i, pid in enumerate(ids):
        terms = goa_lookup.get(pid)
        if not terms:
            continue
        touched += 1
        for go in terms:
            j = term_to_idx.get(go)
            if j is None:
                continue
            old = probs[i, j]
            if mode == "set":
                if old != boost:
                    probs[i, j] = boost
                    updates += 1
            else:  # "max"
                if old < boost:
                    probs[i, j] = boost
                    updates += 1

    return touched, updates





def load_or_build_goa_lookup(
    *,
    gaf_path: str,
    proteins: Iterable[str],
    cache_path: str,
    allowed_aspect: Optional[Set[str]] = None,
):
    """
    Load cached GOA lookup if present; otherwise build and cache it.
    """
    if os.path.exists(cache_path):
        print(f"📦 Loading cached GOA lookup → {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("🔨 Building GOA lookup from GAF (first run only)...")
    goa_lookup = build_goa_lookup_for_proteins(
        gaf_path=gaf_path,
        proteins=proteins,
        allowed_aspect=allowed_aspect,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(goa_lookup, f)

    print(f"💾 Saved GOA lookup → {cache_path}")
    print(f"✔ GOA proteins covered: {len(goa_lookup)}")

    return goa_lookup



###### IF WANT TO INCLUDE NEGATIVE BOOSTING ######
def build_goa_pos_neg_lookup_for_proteins(
    gaf_path: str,
    proteins: Iterable[str],
    allowed_aspect: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    proteins = set(proteins)
    pos: Dict[str, Set[str]] = {}
    neg: Dict[str, Set[str]] = {}

    with _open_text_maybe_gz(gaf_path) as f:
        for line in f:
            if not line or line[0] == "!":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            prot = parts[1]
            if prot not in proteins:
                continue

            qualifier = parts[3]
            go_id = parts[4]
            aspect = parts[8]

            if allowed_aspect is not None and aspect not in allowed_aspect:
                continue

            if "NOT" in qualifier:
                neg.setdefault(prot, set()).add(go_id)
            else:
                pos.setdefault(prot, set()).add(go_id)

    return pos, neg


def load_or_build_goa_pos_neg_lookup(
    *,
    gaf_path: str,
    proteins: Iterable[str],
    cache_path: str,
    allowed_aspect: Optional[Set[str]] = None,
):
    if os.path.exists(cache_path):
        print(f"📦 Loading cached GOA pos/neg → {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("🔨 Building GOA pos/neg from GAF (first run only)...")
    goa_pos, goa_neg = build_goa_pos_neg_lookup_for_proteins(
        gaf_path=gaf_path,
        proteins=proteins,
        allowed_aspect=allowed_aspect,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((goa_pos, goa_neg), f)

    print(f"💾 Saved GOA pos/neg → {cache_path}")
    print(f"✔ GOA pos proteins: {len(goa_pos)} | neg proteins: {len(goa_neg)}")
    return goa_pos, goa_neg

def apply_goa_pos_neg_constraints_inplace(
    ids: Iterable[str],
    probs: np.ndarray,
    vocab: np.ndarray,
    goa_pos: Dict[str, Set[str]],
    goa_neg: Dict[str, Set[str]],
    descendants_map: Optional[Dict[str, Set[str]]] = None,
    neg_cap: float = 1e-4,
    descendant_cap: float = 1e-4,
    descendant_conf_threshold: float = 0.2,
):
    term_to_idx = {str(t): i for i, t in enumerate(vocab)}

    for i, pid in enumerate(ids):
        # Positives -> 1.0
        for go in goa_pos.get(pid, ()):
            j = term_to_idx.get(go)
            if j is not None:
                probs[i, j] = 1.0

        # NOTs -> cap exact term (+ optional descendants)
        for go in goa_neg.get(pid, ()):
            j = term_to_idx.get(go)
            if j is not None:
                probs[i, j] = min(probs[i, j], neg_cap)

            if descendants_map is None:
                continue

            for child in descendants_map.get(go, ()):
                cj = term_to_idx.get(child)
                if cj is None:
                    continue
                if probs[i, cj] < descendant_conf_threshold:
                    probs[i, cj] = min(probs[i, cj], descendant_cap)

