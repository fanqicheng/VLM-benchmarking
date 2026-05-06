import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
import re




def extract_unitopatho_label(slide_name):
    """
    Robust UNITO label extractor.
    Supports:
        - TVA.HG / TVA.LG / TA.HG / TA.LG
        - TVAHG / TVALG / TAHG / TALG
        - TA-HG / TVA_LG
        - Any prefix/suffix around it
    """

    raw = slide_name.upper().replace(" ", "").replace("_", "").replace("-", "")

    CANON = ["TVA.HG", "TVA.LG", "TA.HG", "TA.LG", "HP", "NORM"]

    name_dot = slide_name.upper()
    for c in sorted(CANON, key=lambda x: -len(x)):
        if c in name_dot:
            return c

    dotless_map = {
        "TVAHG": "TVA.HG",
        "TVALG": "TVA.LG",
        "TAHG":  "TA.HG",
        "TALG":  "TA.LG",
    }

    for k, v in dotless_map.items():
        if k in raw:
            return v

    if "HP" in raw:
        return "HP"
    if "NORM" in raw:
        return "NORM"

    raise ValueError(f"[ERROR] Cannot extract UNITO label from slide name: {slide_name}")

def zero_shot_classification(image_embs, text_embs, labels, normalize=True):

    image_embs = torch.tensor(image_embs, dtype=torch.float32)
    text_embs = torch.tensor(text_embs, dtype=torch.float32)

    if normalize:
        image_embs = F.normalize(image_embs, dim=-1)
        text_embs = F.normalize(text_embs, dim=-1)

    sims = image_embs @ text_embs.T
    preds = sims.argmax(dim=1).cpu().numpy()

    labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return preds, sims.cpu().numpy(), acc, f1

def compute_macro_auc(labels, sims, num_classes):
    """
    labels: int array shape (N,)
    sims: similarity scores, shape (N, C)
    """
    try:
        labels_bin = label_binarize(labels, classes=np.arange(num_classes))
        macro_auc = roc_auc_score(
            labels_bin,
            sims,
            average="macro",
            multi_class="ovr"
        )
        return macro_auc
    except Exception as e:
        print("AUC computation error:", e)
        return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_emb", required=True)
    parser.add_argument("--slide_ids", required=True)
    parser.add_argument("--text_emb", required=True)       
    parser.add_argument("--text_labels", required=True)    
    parser.add_argument("--output_dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_embs = np.load(args.image_emb)
    slide_ids = np.load(args.slide_ids)

    text_embs = np.load(args.text_emb)              

    with open(args.text_labels, "r") as f:
        class_names = json.load(f)     

    raw_labels = [extract_unitopatho_label(str(s)) for s in slide_ids]

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    missing = sorted(set(raw_labels) - set(class_names))
    print("Unique raw labels:", sorted(set(raw_labels)))
    print("Class names:", class_names)
    print("Missing labels:", missing)
    
    numeric_labels = np.array([class_to_idx[c] for c in raw_labels], dtype=int)

    preds, sims, acc, f1 = zero_shot_classification(
        image_embs=image_embs,
        text_embs=text_embs,
        labels=numeric_labels,
    )
    C = len(class_names)
    macro_auc = compute_macro_auc(numeric_labels, sims, C)

    print("\n==== UNITO ZERO-SHOT ====")
    print("Accuracy:", acc)
    print("Macro F1:", f1)
    print("Macro AUC:", macro_auc)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
    "accuracy": float(acc),
    "f1_macro": float(f1),
    "auc": None if macro_auc is None else float(macro_auc)
    }

    with open(output_dir / f"results_zeroshot.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()