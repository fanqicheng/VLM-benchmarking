from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

SETS = [
    "set1_openvocab",
    "set2_finegrained",
    "set3_crossorgan",
    "set4_alldatasets",
]

SET_MAX_RECALL = {
    "set1_openvocab":   10,
    "set2_finegrained": 5,
    "set3_crossorgan":  5,
    "set4_alldatasets": 10,
}

DATASET_NAME_MAP = {
    "CAMELYON16":  "CAM16",
    "CAMELYON17":  "CAM17",
    "UNITOPATHO":  "UNITOPATHO",
    "TCGA-GBMLGG": "TCGA-GBMLGG",
    "PANDA":       "PANDA",
}

LABEL_MAPS = {
    "CAMELYON16": {"normal": "NORMAL", "tumor": "TUMOR"},
    "CAMELYON17": {"negative": "NEGATIVE", "itc": "ITC", "micro": "MICRO", "macro": "MACRO"},
    "PANDA":      {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5"},
    "UNITOPATHO": {"norm": "NORM", "hp": "HP", "ta.lg": "TA.LG", "ta.hg": "TA.HG", "tva.lg": "TVA.LG", "tva.hg": "TVA.HG"},
    "TCGA-GBMLGG": {"lgg": "LGG", "gbm": "GBM"},
}


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def extract_unitopatho_label(slide_name: str) -> str:
    name     = slide_name.upper().replace(" ", "").replace("_", "").replace("-", "")
    name_dot = slide_name.upper()
    CANONICAL = ["TVA.HG", "TVA.LG", "TA.HG", "TA.LG", "HP", "NORM"]
    for lbl in CANONICAL:
        if lbl in name_dot:
            return lbl
    DOTLESS = {"TVAHG": "TVA.HG", "TVALG": "TVA.LG", "TAHG": "TA.HG", "TALG": "TA.LG"}
    for k, v in DOTLESS.items():
        if k in name:
            return v
    if "HP" in name: return "HP"
    if "NORM" in name: return "NORM"
    raise ValueError(f"Cannot extract UNITOPATHO label from: {slide_name}")


def align_labels(slide_ids: np.ndarray, csv_path, dataset: str):
    """Returns (gt_labels, valid_indices) where gt_labels are label keys like 'NORMAL', 'TUMOR'."""
    label_map = LABEL_MAPS[dataset]

    if dataset == "UNITOPATHO":
        gt, valid_indices = [], []
        for i, sid in enumerate(slide_ids):
            lbl = extract_unitopatho_label(str(sid))
            gt.append(label_map[lbl.lower()])
            valid_indices.append(i)
        return np.array(gt), np.array(valid_indices)

    df = pd.read_csv(csv_path)

    if dataset == "CAMELYON17":
        df = df[df["patient"].str.contains(".tif")].copy()
        df["patient"] = df["patient"].apply(lambda x: Path(x).stem.lower())
        id_to_label = dict(zip(df["patient"], df["stage"].str.lower()))
    elif dataset == "TCGA-GBMLGG":
        id_to_label = {}
        for _, row in df.iterrows():
            case = str(row["casename"])
            raw  = int(row["idh mutation"])
            id_to_label[case] = "gbm" if raw == 0 else "lgg"
    elif dataset == "PANDA":
        id_to_label = dict(zip(df["image_id"].astype(str), df["isup_grade"].astype(str)))
    else:  # CAMELYON16
        id_to_label = {}
        for _, row in df.iterrows():
            sid = str(row["image"]).replace(".tif", "")
            id_to_label[sid] = str(row["type"]).lower()

    gt, valid_indices = [], []
    for i, sid in enumerate(slide_ids):
        if dataset == "CAMELYON17":
            key = Path(str(sid)).stem.lower()
        elif dataset == "TCGA-GBMLGG":
            key = "-".join(str(sid).split("-")[:3])
        else:
            key = str(sid).replace(".tif", "")
        if key not in id_to_label:
            continue
        gt.append(label_map[id_to_label[key]])
        valid_indices.append(i)

    return np.array(gt), np.array(valid_indices)


def compute_mrr(ranks: list) -> float:
    return float(np.mean([1.0 / r for r in ranks]))


def compute_ndcg(ranks: list) -> float:
    return float(np.mean([1.0 / np.log2(r + 1) for r in ranks]))


def evaluate_one_set(
    image_emb:  np.ndarray,
    slide_ids:  np.ndarray,
    gt_labels:  np.ndarray,
    emb_dir:    Path,
    dataset:    str,
    max_k:      int = 5,
) -> dict:
    """
    Each slide is evaluated against its own label-specific candidate pool.
    """
    file_dataset = DATASET_NAME_MAP.get(dataset, dataset)  # CAM16, CAM17, etc.
    ground_truths = json.loads((emb_dir / f"{file_dataset}_ground_truth.json").read_text())

    label_cache = {}
    for lbl in set(gt_labels):
        cand_path  = emb_dir / f"{file_dataset}_{lbl}_candidates.npy"
        texts_path = emb_dir / f"{file_dataset}_{lbl}_candidate_texts.json"
        if not cand_path.exists():
            raise FileNotFoundError(f"Missing: {cand_path}")
        label_cache[lbl] = {
            "emb":   normalize(np.load(cand_path)),
            "texts": json.loads(texts_path.read_text()),
        }

    correct_at_k = {k: 0 for k in range(1, max_k + 1)}
    ranks = []

    for i, (sid, lbl) in enumerate(zip(slide_ids, gt_labels)):
        cand_emb   = label_cache[lbl]["emb"]    
        cand_texts = label_cache[lbl]["texts"]
        gt_text    = ground_truths[lbl]

        sims         = image_emb[i] @ cand_emb.T   
        order        = np.argsort(sims)[::-1]
        ranked_texts = [cand_texts[j] for j in order]

        if gt_text in ranked_texts:
            rank = ranked_texts.index(gt_text) + 1
        else:
            rank = len(cand_texts) + 1

        ranks.append(rank)
        
        for k in correct_at_k:
            if rank <= k:
                correct_at_k[k] += 1

    total = len(slide_ids)

    n_cands = {lbl: len(label_cache[lbl]["texts"]) for lbl in label_cache}

    return {
        **{f"recall@{k}": round(correct_at_k[k] / total, 4) for k in range(1, max_k + 1)},
        "MRR":            round(compute_mrr(ranks), 4),
        "nDCG":           round(compute_ndcg(ranks), 4),
        "n_slides":       total,
        "n_candidates":   n_cands,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Open vocabulary retrieval evaluation")
    parser.add_argument("--image_emb",  required=True,  help="Image embeddings .npy [N_slides, D]")
    parser.add_argument("--slide_ids",  required=True,  help="Slide IDs .npy")
    parser.add_argument("--emb_root",   required=True,  help="Root dir of text embeddings")
    parser.add_argument("--csv_path",   required=False, default=None, help="Ground truth CSV")
    parser.add_argument("--dataset",    required=True,  help="Dataset name (e.g. CAMELYON16)")
    parser.add_argument("--model",      required=True,  help="Model name (e.g. plip)")
    parser.add_argument("--sets",       nargs="+",      default=None, help="Subset of sets (default: all)")
    parser.add_argument("--output_dir", default="./results/open_vocab")
    return parser.parse_args()


def main():
    args = parse_args()

    image_emb = normalize(np.load(args.image_emb))
    slide_ids = np.load(args.slide_ids)

    gt_labels, valid_indices = align_labels(
        slide_ids,
        Path(args.csv_path) if args.csv_path else None,
        args.dataset
    )
    image_emb = image_emb[valid_indices]
    slide_ids = slide_ids[valid_indices]

    sets_to_run = args.sets or SETS
    emb_root    = Path(args.emb_root)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Model   : {args.model}")
    print(f"  Slides  : {len(slide_ids)}")
    print(f"{'='*60}\n")

    all_results = {}

    for set_name in sets_to_run:
        emb_dir = emb_root / args.model / set_name
        if not emb_dir.exists():
            print(f"Skipping {set_name}: {emb_dir} not found")
            continue
        
        max_k = SET_MAX_RECALL.get(set_name, 5) 

        print(f"── {set_name} ──")
        try:
            metrics = evaluate_one_set(image_emb, slide_ids, gt_labels, emb_dir, args.dataset, max_k)
            all_results[set_name] = metrics
            for k in range(1, max_k + 1):
                print(f"   recall@{k} : {metrics[f'recall@{k}']}")
            print(f"   MRR      : {metrics['MRR']}")
            print(f"   nDCG     : {metrics['nDCG']}")
        except Exception as e:
            print(f"Error: {e}")
        print()

    out_path = out_dir / f"{args.dataset}_{args.model}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "model":   args.model,
        "results": all_results,
    }, indent=2))

    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()