from goatools.obo_parser import GODag
import math
from collections import defaultdict
import numpy as np
import pickle
import pandas as pd
import gc
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
############################################


CONFIG = {
    "train_terms_path": r"cafa-6-protein-function-prediction\Train\train_terms.tsv",
    "train_ids_path": "train_ids.npy",
    "obo_path": r"cafa-6-protein-function-prediction\Train\go-basic.obo",
    "N_labels": 1024,
}

############################################
# LOAD GO DAG
############################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_go_dag(path):
    print("📦 Loading GO DAG...")
    go = GODag(path)
    print("✔ Loaded", len(go), "GO terms")
    return go


def load_ic_from_file(path):
    """
    Load GO → IC weights from IA.tsv file.
    Returns dict: {GO_ID: IC_value}
    """
    ic = {}
    with open(path, "r") as f:
        for line in f:
            go, val = line.strip().split("\t")
            ic[go] = float(val)
    return ic


############################################
# PROPAGATION + IC
############################################

def propagate_terms(term_set, go_dag, aspect=None):
    out = set()

    for go_id in term_set:
        if go_id not in go_dag:
            continue

        term = go_dag[go_id]

        # filter by MF/BP/CC
        if aspect and term.namespace[0].upper() != aspect:
            continue

        # include the term itself
        out.add(go_id)

        # get_all_parents() returns GO *IDs*, not GOterm objects
        parent_ids = term.get_all_parents()

        # If restricting by MF/BP/CC, filter
        if aspect:
            for pid in parent_ids:
                if pid in go_dag and go_dag[pid].namespace[0].upper() == aspect:
                    out.add(pid)
        else:
            out.update(parent_ids)

    return out



def compute_ic_weights(y_true, go_dag, aspect, cache_file=None):
    """
    Computes IC weights for GO terms with tqdm progress bar.
    Optionally loads/saves cache to avoid recomputation.
    """

    # --------------------
    # Load from cache?
    # --------------------
    if cache_file and os.path.exists(cache_file):
        print(f"🔁 Loading cached IC weights from {cache_file}")
        return pickle.load(open(cache_file, "rb"))

    print("📊 Computing Information Content (IC) weights...")
    term_counts = defaultdict(int)

    # --------------------
    # Count GO frequencies with propagation
    # --------------------
    for pid, terms in tqdm(y_true.items(), desc="Propagating terms", ncols=80):
        if not terms:
            continue

        propagated = propagate_terms(terms, go_dag, aspect)

        for t in propagated:
            term_counts[t] += 1

    total = len(y_true)

    # --------------------
    # Compute IC = -log(p)
    # --------------------
    ic = {}
    for go, count in term_counts.items():
        p = count / total
        ic[go] = -math.log(p) if p > 0 else 0.0

    # --------------------
    # Save to cache?
    # --------------------
    if cache_file:
        print(f"💾 Saving IC weights to {cache_file}")
        pickle.dump(ic, open(cache_file, "wb"))
    return ic


############################################
# HIERARCHY FIXING
############################################

def make_hierarchy_consistent(scores, go_dag, aspect):
    new_scores = dict(scores)
    terms = [go for go in scores if go in go_dag]
    terms.sort(key=lambda go: go_dag[go].depth, reverse=True)

    for go in terms:
        s = new_scores.get(go, 0.0)
        for parent in go_dag[go].parents:
            pid = parent.id
            if aspect and go_dag[pid].namespace[0].upper() != aspect:
                continue
            if new_scores.get(pid, 0.0) < s:
                new_scores[pid] = s
    return new_scores


############################################
# WEIGHTED HIERARCHICAL FMAX
############################################

from tqdm import tqdm
import numpy as np

def weighted_hierarchical_fmax(y_true, y_scores, go_dag, ic_weights, aspect):
    thresholds = np.linspace(0, 1, 41)   # step 0.025 for speed

    # Propagate true sets
    true_prop = {
        pid: propagate_terms(tset, go_dag, aspect)
        for pid, tset in tqdm(y_true.items(), desc="True propagation", ncols=80)
    }

    # Hierarchy-consistent predictions
    scores_hc = {
        pid: make_hierarchy_consistent(scores, go_dag, aspect)
        for pid, scores in tqdm(y_scores.items(), desc="Hierarchy fix", ncols=80)
    }

    proteins = sorted(set(true_prop.keys()) & set(scores_hc.keys()))

    best_F = -1
    best_t = None

    curve = []  # store precision, recall, F

    print("Scanning thresholds...")
    for t in tqdm(thresholds, ncols=80):
        TPw = FPw = FNw = 0.0

        for pid in proteins:
            true_set = true_prop[pid]

            pred_raw = {go for go, s in scores_hc[pid].items() if s >= t}
            pred_set = propagate_terms(pred_raw, go_dag, aspect)

            TPw += sum(ic_weights.get(go, 0.0) for go in pred_set & true_set)
            FPw += sum(ic_weights.get(go, 0.0) for go in pred_set - true_set)
            FNw += sum(ic_weights.get(go, 0.0) for go in true_set - pred_set)

        P = TPw / (TPw + FPw) if TPw + FPw > 0 else 0.0
        R = TPw / (TPw + FNw) if TPw + FNw > 0 else 0.0
        F = 2 * P * R / (P + R) if P + R > 0 else 0.0

        curve.append({
            "threshold": float(t),
            "precision": P,
            "recall": R,
            "F": F,
        })

        if F > best_F:
            best_F = F
            best_t = t

    return {
        "Fmax_weighted": best_F,
        "threshold": best_t,
        "curve": curve
    }



from tqdm import tqdm
import numpy as np

def build_y_scores(model_outputs, go_terms_pred, mf_idx, bp_idx, cc_idx):
    y_scores = {}
    y_scores_mf = {}
    y_scores_bp = {}
    y_scores_cc = {}

    for pid, vec in tqdm(model_outputs.items(), desc="Building y_scores", ncols=80):
        vec = np.asarray(vec).reshape(-1)

        # convert logits → probabilities
        probs = sigmoid(vec)

        # full set
        y_scores[pid] = {go_terms_pred[i]: float(probs[i]) for i in range(len(go_terms_pred))}

        # MF
        y_scores_mf[pid] = {go_terms_pred[i]: float(probs[i]) for i in mf_idx}

        # BP
        y_scores_bp[pid] = {go_terms_pred[i]: float(probs[i]) for i in bp_idx}

        # CC
        y_scores_cc[pid] = {go_terms_pred[i]: float(probs[i]) for i in cc_idx}

    return y_scores, y_scores_mf, y_scores_bp, y_scores_cc



def generate_dummy_model_outputs(protein_ids, go_terms_pred, seed=0):
    """
    Create dummy model outputs for testing the evaluation pipeline.

    Parameters
    ----------
    protein_ids : list[str]
        List of protein IDs (same PIDs used in metadata)

    go_terms_pred : list[str]
        List of GO terms predicted by the model (length 2048)

    seed : int
        Random seed for reproducibility

    Returns
    -------
    model_outputs : dict
        model_outputs[pid] = np.array shape (2048,) with values in 0–1
    """
    np.random.seed(seed)

    N = len(go_terms_pred)
    model_outputs = {}

    for pid in protein_ids:
        # Random probabilities for each of the 2048 GO terms
        vec = np.random.rand(N).astype(np.float32)
        model_outputs[pid] = vec

    return model_outputs

def plot_pr_curves(results_dict, save_path=None, title="Precision–Recall Curve"):
    """
    results_dict: {
        "MF": res_mf,
        "BP": res_bp,
        "CC": res_cc
    }
    """
    plt.figure(figsize=(7,6))

    for label, result in results_dict.items():
        curve = result["curve"]
        P = [c["precision"] for c in curve]
        R = [c["recall"]    for c in curve]
        plt.plot(R, P, marker="o", label=label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_fmax_curves(results_dict, save_path=None, title="Threshold vs F-score"):
    plt.figure(figsize=(7,6))

    for label, result in results_dict.items():
        curve = result["curve"]
        thresholds = [c["threshold"] for c in curve]
        F_scores   = [c["F"] for c in curve]
        plt.plot(thresholds, F_scores, marker="o", label=label)

    plt.xlabel("Threshold")
    plt.ylabel("Weighted F-score")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

import pickle
import os

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✔ Saved pickle to {path}")
############################################
# MAIN LOGIC
############################################

if __name__ == "__main__":

    # Load metadata
    metadata = pickle.load(open("t5_metadata.pkl", "rb"))
    fold0 = pd.read_csv("splits/fold_0.tsv", sep="\t")
    pids_ordered = fold0["protein"].tolist()

   
    # Load raw GO annotations
    terms_df = pd.read_csv(CONFIG["train_terms_path"], sep="\t")

    # Compute top-2048 most common GO terms
    top_terms = (
        terms_df["term"]
        .value_counts()
        .index[:CONFIG["N_labels"]]
        .tolist()
    )

    go_terms_pred = top_terms
    go_set_pred = set(go_terms_pred)

    # Build filtered true labels
    y_true = {
        pid: (set(info["labels"]["MF"] + info["labels"]["BP"] + info["labels"]["CC"]) 
            & go_set_pred)
        for pid, info in metadata.items()
    }

    y_true_mf = {pid: set(info["labels"]["MF"]) & go_set_pred for pid, info in metadata.items()}
    y_true_bp = {pid: set(info["labels"]["BP"]) & go_set_pred for pid, info in metadata.items()}
    y_true_cc = {pid: set(info["labels"]["CC"]) & go_set_pred for pid, info in metadata.items()}

    print("MF: proteins with at least one valid label:",
      sum(len(v)>0 for v in y_true_mf.values()))

    print("BP: proteins with at least one valid label:",
        sum(len(v)>0 for v in y_true_bp.values()))

    print("CC: proteins with at least one valid label:",
        sum(len(v)>0 for v in y_true_cc.values()))

    print("Total proteins:", len(y_true_mf))
    


    

    preds = np.load(r"fold_0_logits.npy/fold_0_logits.npy")

    print(go_terms_pred[:5])
    print(list(y_true_mf)[:5])
    print([g for g,_ in preds[:5]])


    model_outputs = {
    pids_ordered[i]: preds[i]
    for i in range(len(pids_ordered)-1)
}
    del metadata
    gc.collect()
    
   
     # Load GO DAG
    go_dag = load_go_dag(CONFIG["obo_path"])

    # Load IA.tsv
    ic_all = load_ic_from_file(r"cafa-6-protein-function-prediction\IA.tsv")

 
    # IC values loaded from IA.tsv, restricted to prediction vocabulary
    ic_pred = {go: ic_all.get(go, 0.0) for go in go_terms_pred}

    # MF
    ic_mf = {
        go: ic_pred.get(go, 0.0)
        for go in go_terms_pred
        if go in go_dag and go_dag[go].namespace == "molecular_function"
    }

    # BP
    ic_bp = {
        go: ic_pred.get(go, 0.0)
        for go in go_terms_pred
        if go in go_dag and go_dag[go].namespace == "biological_process"
    }

    # CC
    ic_cc = {
        go: ic_pred.get(go, 0.0)
        for go in go_terms_pred
        if go in go_dag and go_dag[go].namespace == "cellular_component"
    }


        # first compute MF/BP/CC indices
    mf_idx = [i for i, go in enumerate(go_terms_pred) if go in go_dag and go_dag[go].namespace == "molecular_function"]
    bp_idx = [i for i, go in enumerate(go_terms_pred) if go in go_dag and go_dag[go].namespace == "biological_process"]
    cc_idx = [i for i, go in enumerate(go_terms_pred) if go in go_dag and go_dag[go].namespace == "cellular_component"]

    print("MF idx count:", len(mf_idx))
    print("BP idx count:", len(bp_idx))
    print("CC idx count:", len(cc_idx))



    # print(bp_idx[:10])
    y_scores, y_scores_mf, y_scores_bp, y_scores_cc = build_y_scores(
        model_outputs,
        go_terms_pred,
        mf_idx, bp_idx, cc_idx
    )


    # Restrict evaluation to proteins that have BOTH truth and predictions
    eval_pids = set(y_scores_mf.keys()) & set(y_true_mf.keys())

    # Filter truth to eval set
    y_true_mf = {pid: y_true_mf[pid] for pid in eval_pids}
    y_true_bp = {pid: y_true_bp[pid] for pid in eval_pids}
    y_true_cc = {pid: y_true_cc[pid] for pid in eval_pids}

    # Filter predictions to eval set
    y_scores_mf = {pid: y_scores_mf[pid] for pid in eval_pids}
    y_scores_bp = {pid: y_scores_bp[pid] for pid in eval_pids}
    y_scores_cc = {pid: y_scores_cc[pid] for pid in eval_pids}

    print("Evaluation proteins:", len(eval_pids))

    # Check MF prediction magnitudes
    mf_max = max(max(y_scores_mf[pid].values()) for pid in y_scores_mf)
    bp_max = max(max(y_scores_bp[pid].values()) for pid in y_scores_bp)
    cc_max = max(max(y_scores_cc[pid].values()) for pid in y_scores_cc)

    print("Max MF prob:", mf_max)
    print("Max BP prob:", bp_max)
    print("Max CC prob:", cc_max)

    save_pickle(y_scores,    "scores/y_scores_full.pkl")
    save_pickle(y_scores_mf, "scores/y_scores_MF.pkl")
    save_pickle(y_scores_bp, "scores/y_scores_BP.pkl")
    save_pickle(y_scores_cc, "scores/y_scores_CC.pkl")

    print("MF truth present in eval subset:",
      sum(len(v)>0 for v in y_true_mf.values()))

    print("BP truth present in eval subset:",
        sum(len(v)>0 for v in y_true_bp.values()))

    print("CC truth present in eval subset:",
        sum(len(v)>0 for v in y_true_cc.values()))


    print(y_scores_mf[list(y_scores_mf.keys())[0]])
   
    # Evaluate MF
    res_mf = weighted_hierarchical_fmax(
        y_true_mf,
        y_scores_mf,
        go_dag,
        ic_mf,
        aspect="F"
    )

    print("MF Weighted Fmax:", res_mf["Fmax_weighted"])
    


    res_bp = weighted_hierarchical_fmax(
        y_true_bp,
        y_scores_bp,
        go_dag,
        ic_bp,
        aspect="P"
    )
    print("BP Weighted Fmax:", res_bp["Fmax_weighted"])

    res_cc = weighted_hierarchical_fmax(
        y_true_cc,
        y_scores_cc,
        go_dag,
        ic_cc,
        aspect="C")
    

    print("CC Weighted Fmax:", res_cc["Fmax_weighted"])


    plot_fmax_curves(
        {"MF": res_mf, "BP": res_bp, "CC": res_cc},
        save_path="fmax_curve.png")
   

