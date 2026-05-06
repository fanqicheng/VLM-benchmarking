from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

LABEL_MAPS = {
    "CAMELYON16": {"normal": 0, "tumor": 1},
    "CAMELYON17": {"negative": 0, "itc": 1, "micro": 2, "macro": 3},
    "UNITOPATHO": {"norm": 0, "hp": 1, "ta.lg": 2, "ta.hg": 3, "tva.lg": 4, "tva.hg": 5},
    "TCGA-GBMLGG": {"lgg": 0, "gbm": 1},
    "PANDA": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
}

def extract_unitopatho_label(slide_name: str) -> str:
    name = slide_name.upper().replace(" ", "").replace("_", "").replace("-", "")
    name_dot = slide_name.upper()
    CANONICAL = ["TVA.HG", "TVA.LG", "TA.HG", "TA.LG", "HP", "NORM"]
    for lbl in CANONICAL:
        if lbl in name_dot:
            return lbl
    DOTLESS = {"TVAHG": "TVA.HG", "TVALG": "TVA.LG", "TAHG": "TA.HG", "TALG": "TA.LG"}
    for k, v in DOTLESS.items():
        if k in name:
            return v
    if "HP" in name:
        return "HP"
    if "NORM" in name:
        return "NORM"
    raise ValueError(f"Cannot extract UNITOPATHO label from: {slide_name}")


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def align_labels(slide_ids: np.ndarray, csv_path, dataset: str):
    label_map = LABEL_MAPS[dataset]

    if dataset == "UNITOPATHO":
        gt, valid_indices = [], []
        for i, sid in enumerate(slide_ids):
            lbl = extract_unitopatho_label(str(sid))
            gt.append(label_map[lbl.lower()])
            valid_indices.append(i)
        return np.array(gt, dtype=int), np.array(valid_indices)

    df = pd.read_csv(csv_path)

    if dataset in ("CAM17", "CAMELYON17"):
        df = df[df["patient"].str.contains(".tif")].copy()
        df["patient"] = df["patient"].apply(lambda x: Path(x).stem.lower())
        id_to_label = dict(zip(df["patient"], df["stage"].str.lower()))
    elif dataset == "TCGA-GBMLGG":
        id_to_label = {}
        for _, row in df.iterrows():
            case = str(row["casename"])
            raw = int(row["idh mutation"])
            id_to_label[case] = "gbm" if raw == 0 else "lgg"
    elif dataset == "PANDA":
        id_to_label = dict(zip(df["image_id"].astype(str), df["isup_grade"].astype(str)))
    else:
        id_to_label = {}
        for _, row in df.iterrows():
            sid = str(row["image"]).replace(".tif", "")
            id_to_label[sid] = str(row["type"]).lower()

    gt, valid_indices = [], []
    for i, sid in enumerate(slide_ids):
        if dataset in ("CAM17", "CAMELYON17"):
            key = Path(str(sid)).stem.lower()
        elif dataset == "TCGA-GBMLGG":
            key = "-".join(str(sid).split("-")[:3])
        else:
            key = str(sid).replace(".tif", "")
        if key not in id_to_label:
            continue
        gt.append(label_map[id_to_label[key]])
        valid_indices.append(i)

    return np.array(gt, dtype=int), np.array(valid_indices)

def compute_alignment_and_gap(image_emb: np.ndarray, text_emb: np.ndarray, gt: np.ndarray) -> dict:
    """
    Alignment Score and Similarity Gap as defined in Guo et al. (MELBA 2025).

    Args:
        image_emb : [N, D] normalized image embeddings
        text_emb  : [C, D] normalized text embeddings (one per class)
        gt        : [N] ground truth class indices

    Returns:
        dict with alignment and similarity_gap
    """
    image_emb = normalize(image_emb)
    text_emb  = normalize(text_emb)

    N = len(gt)
    C = text_emb.shape[0]

    sims = image_emb @ text_emb.T 

    pos_sim = np.array([sims[i, gt[i]] for i in range(N)])

    neg_sim = np.array([
        (sims[i].sum() - sims[i, gt[i]]) / (C - 1)
        for i in range(N)
    ])

    alignment      = float(np.mean(pos_sim))
    similarity_gap = float(np.mean(pos_sim - neg_sim))

    return {
        "alignment":      round(alignment, 4),
        "similarity_gap": round(similarity_gap, 4),
        "n_slides":       N,
        "n_classes":      C,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Compute image-text alignment score and similarity gap")
    parser.add_argument("--image_emb",  required=True,  help="Path to image embeddings .npy [N, D]")
    parser.add_argument("--slide_ids",  required=True,  help="Path to slide ids .npy")
    parser.add_argument("--text_emb",   required=True,  help="Path to text embeddings .npy [C, D]")
    parser.add_argument("--csv_path",   required=False, default=None, help="Path to ground truth CSV")
    parser.add_argument("--dataset",    required=True,  help="Dataset name (e.g. CAMELYON16)")
    parser.add_argument("--model",      required=True,  help="Model name (e.g. plip)")
    parser.add_argument("--output_dir", default="./results/alignment")
    return parser.parse_args()


def main():
    args = parse_args()

    image_emb = np.load(args.image_emb)
    slide_ids = np.load(args.slide_ids)
    text_emb  = np.load(args.text_emb)

    gt, valid_indices = align_labels(
        slide_ids,
        args.csv_path,
        args.dataset
    )
    image_emb = image_emb[valid_indices]

    print(f"\n{'='*50}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Model   : {args.model}")
    print(f"  Slides  : {len(gt)}")
    print(f"  Classes : {text_emb.shape[0]}")
    print(f"{'='*50}\n")

    results = compute_alignment_and_gap(image_emb, text_emb, gt)

    print(f"  Alignment      : {results['alignment']}")
    print(f"  Similarity Gap : {results['similarity_gap']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.model}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "model":   args.model,
        **results,
    }, indent=2))

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()