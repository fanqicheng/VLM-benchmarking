import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import json


def compute_similarity(img_embs, txt_embs):
    img = torch.tensor(img_embs, dtype=torch.float32)
    txt = torch.tensor(txt_embs, dtype=torch.float32)
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    return (img @ txt.T).cpu().numpy()

def compute_mrr(ranks):
    return np.mean([1.0 / r for r in ranks])

def compute_ndcg(ranks):
    return np.mean([1.0 / np.log2(r + 1) for r in ranks])

def extract_unitopatho_label(slide_name):
    """
        TA.HG / TAHG
        TA.LG / TALG
        TVA.HG / TVAHG
        TVA.LG / TVALG
        HP, NORM
    """
    name = slide_name.upper().replace(" ", "").replace("_", "").replace("-", "")
    name_dot = slide_name.upper()

    CANONICAL = ["TVA.HG", "TVA.LG", "TA.HG", "TA.LG", "HP", "NORM"]

    for lbl in CANONICAL:
        if lbl in name_dot:
            return lbl

    DOTLESS = {
        "TVAHG": "TVA.HG",
        "TVALG": "TVA.LG",
        "TAHG": "TA.HG",
        "TALG": "TA.LG",
    }
    for k, v in DOTLESS.items():
        if k in name:
            return v

    if "HP" in name:
        return "HP"
    if "NORM" in name:
        return "NORM"

    raise ValueError(f"Cannot extract UNITOPATHO label from: {slide_name}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True) # camelyon17
    p.add_argument("--image_emb", required=True)
    p.add_argument("--slide_ids", required=True)

    # distractor folder
    p.add_argument("--text_root", required=True)

    # ALL 4 datasets now use CSV
    p.add_argument("--csv_path", required=False)

    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out_json", required=True)
    return p.parse_args()

def load_ground_truth(dataset, slide_ids, csv_path):
    dataset = dataset.upper()

    if dataset == "UNITOPATHO":
        return np.array([
            extract_unitopatho_label(str(sid))
            for sid in slide_ids
        ])

    df = pd.read_csv(csv_path)

    if dataset == "PANDA":
        id2label = dict(zip(df["image_id"], df["isup_grade"]))
        return np.array([str(id2label[s]) for s in slide_ids])

    elif dataset == "CAMELYON17":
        df = pd.read_csv(csv_path)

        # Keep ONLY node-level rows (contain .tif)
        df_nodes = df[df["patient"].str.contains(".tif")].copy()

        # Normalize keys: patient_000_node_0 (no extension)
        df_nodes["patient"] = df_nodes["patient"].apply(lambda x: Path(x).stem.lower())

        id2label = dict(zip(df_nodes["patient"], df_nodes["stage"]))

        # Filter to match only valid image embeddings
        filtered_labels = []
        filtered_ids = []
        filtered_embs = []

        img_embs = load_ground_truth.image_embs   # already provided from main()

        for emb, sid in zip(img_embs, slide_ids):
            key = Path(str(sid)).stem.lower()   # e.g., patient_003_node_4

            if key in id2label:
                filtered_embs.append(emb)
                filtered_ids.append(sid)
                filtered_labels.append(id2label[key].lower())  # negative/micro/macro/itc

        return np.array(filtered_labels), np.array(filtered_ids), np.array(filtered_embs)

    elif dataset == "TCGA-GBMLGG":
        id2label = dict(zip(df["casename"], df["idh mutation"]))

        gt = []
        valid_slide_ids = []
        valid_embs = []

        img_embs = load_ground_truth.image_embs

        for emb, sid in zip(img_embs, slide_ids):
            case = "-".join(sid.split("-")[:3])  # TCGA-02-0004

            if case not in id2label:
                continue

            # Map numeric label → string label
            raw = id2label[case]
            if raw == 0:
                mapped = "GBM"
            elif raw == 1:
                mapped = "LGG"
            else:
                raise ValueError(f"Unexpected label {raw} for {case}")

            valid_slide_ids.append(sid)
            valid_embs.append(emb)
            gt.append(mapped)

        return np.array(gt), np.array(valid_slide_ids), np.array(valid_embs)

    elif dataset == "CAMELYON16":
        df = pd.read_csv(csv_path)

        # normalize CSV image column
        df["image"] = df["image"].astype(str)

        id2label = dict(zip(df["image"], df["type"]))

        valid_labels = []
        valid_ids = []
        valid_embs = []

        img_embs = load_ground_truth.image_embs

        def normalize_id(x):
            x = str(x)
            if not x.endswith(".tif"):
                x = x + ".tif"
            return x

        for emb, sid in zip(img_embs, slide_ids):
            sid_norm = normalize_id(sid)

            if sid_norm not in id2label:
                continue

            label = id2label[sid_norm].lower()  # normal / tumor

            valid_ids.append(sid)
            valid_embs.append(emb)
            valid_labels.append(label)

        print(f"[CAM16] matched: {len(valid_labels)} samples")

        return np.array(valid_labels), np.array(valid_ids), np.array(valid_embs)

def main():
    args = parse_args()
    dataset = args.dataset.upper()

    print("🔵 Loading image embeddings...")
    image_embs = np.load(args.image_emb)
    slide_ids = np.load(args.slide_ids)

    text_dir = Path(args.text_root) / f'{dataset}_distractor'
    print("🔵 Loading text embeddings from:", text_dir)

    text_embs = np.load(text_dir / "text_emb.npy")
    text_descs = np.load(text_dir / "text_desc.npy")
    text_classes = np.load(text_dir / "text_class.npy").astype(str)
    text_classes = np.char.lower(text_classes)

    print(f"[INFO] Loaded {len(text_embs)} text candidates for {dataset}")

    print("🔵 Loading ground truth labels...")
    load_ground_truth.image_embs = image_embs

    res = load_ground_truth(dataset, slide_ids, args.csv_path)

    if isinstance(res, tuple):
        gt_labels, slide_ids, image_embs = res
    else:
        gt_labels = res

    if dataset == 'TCGA-GBMLGG':
        gt_labels = np.char.lower(gt_labels)
        text_classes = np.char.lower(text_classes)

    if dataset == "UNITOPATHO":
        gt_labels = np.char.lower(gt_labels.astype(str))

    print("🔵 Computing cosine similarity...")
    sims = compute_similarity(image_embs, text_embs)

    correct1, correct5 = 0, 0
    ranks = []
    results = {}

    for i, sid in enumerate(slide_ids):
        label = gt_labels[i]
        sim_row = sims[i]

        order = np.argsort(sim_row)[::-1]
        topk = order[:args.topk]
        retrieved_classes = text_classes[order]

        match_positions = np.where(retrieved_classes == label)[0]
        rank = match_positions[0] + 1 if len(match_positions) > 0 else 9999
        ranks.append(rank)

        if text_classes[topk[0]] == label:
            correct1 += 1

        if label in text_classes[topk]:
            correct5 += 1

        recall1_i = int(text_classes[topk[0]] == label)
        recall5_i = int(label in text_classes[topk])
        mrr_i = 1.0 / rank if rank != 9999 else 0.0
        ndcg_i = 1.0 / np.log2(rank + 1) if rank != 9999 else 0.0

        results[str(i)] = {
            "slide_id": sid,
            "gt_label": label,

            "top1_class": text_classes[topk[0]],
            "top1_score": float(sim_row[topk[0]]),

            "rank": int(rank),

            "recall@1": recall1_i,
            "recall@5": recall5_i,
            "mrr": mrr_i,
            "ndcg": ndcg_i,

            "retrieved": [
                {
                    "text": str(text_descs[j]),
                    "text_class": text_classes[j],
                    "sim": float(sim_row[j])
                }
                for j in topk
            ]
        }

    total = len(slide_ids)
    recall1 = correct1 / total
    recall5 = correct5 / total
    mrr = compute_mrr(ranks)
    ndcg = compute_ndcg(ranks)

    print("\n========== RESULTS ==========")
    print("Recall@1:", recall1)
    print("Recall@5:", recall5)
    print("MRR:", mrr)
    print("nDCG:", ndcg)

    with open(args.out_json, "w") as f:
        json.dump({
            "results": results,
            "metrics": {
                "recall@1": recall1,
                "recall@5": recall5,
                "MRR": mrr,
                "nDCG": ndcg
            }
        }, f, indent=4)

    print("Saved →", args.out_json)



if __name__ == "__main__":
    main()