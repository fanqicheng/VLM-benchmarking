import numpy as np
import argparse
import pandas as pd
import re
from pathlib import Path


def get_positive_indices(text_classes, n_per_class=4):
    classes_seen = {}
    indices = []
    for i, cls in enumerate(text_classes):
        if cls not in classes_seen:
            classes_seen[cls] = 0
        if classes_seen[cls] < n_per_class:
            indices.append(i)
            classes_seen[cls] += 1
    return indices

def extract_tcga_case(text):
    m = re.match(r"(TCGA-\d{2}-\d{4})", text)
    return m.group(1) if m else None

def extract_unitopatho_label(name):
    """
    Robust UNITOPATHO label parser.
    Handles all known formats:
      - TVA.HG / TVA.LG / TA.HG / TA.LG
      - TVAHG / TVALG / TAHG / TALG
      - HP / NORM
      - TA.HG CASO 1, 100 B2 TVALG, etc.
    """

    raw = name.upper().replace(" ", "").replace("_", "").replace("-", "")

    mapping = {
        "TVAHG": "TVA.HG",
        "TVALG": "TVA.LG",
        "TAHG":  "TA.HG",
        "TALG":  "TA.LG",
        "HP":    "HP",
        "NORM":  "NORM",
    }

    for key, val in mapping.items():
        if key in raw:
            return val

    original = name.upper()

    for key, val in mapping.items():
        if key in original.replace(" ", "").replace("_", "").replace("-", ""):
            return val
        if key.replace(".", "") in original.replace(".", ""):
            return val

    tokens = original.replace("-", " ").replace("_", " ").split()
    for t in tokens:
        t_clean = t.replace(".", "").upper()
        for key, val in mapping.items():
            if t_clean == key:
                return val
            if t_clean.startswith(key):
                return val

    tail = raw[-6:]  
    for key, val in mapping.items():
        if tail.endswith(key):
            return val

    raise ValueError(f"[UNITO parser] Cannot parse label from slide name: {name}")


def extract_panda_id(name):
    return str(name)


def extract_cam17_id(name):
    return f"{name}.tif"

def cosine_similarity(query, db):
    q = query / np.linalg.norm(query)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    return db @ q

def match(gt, label, dataset):
    """Prefix matching: TA matches TA.HG / TA.LG, etc."""
    if dataset == 'unitopatho':
        gt_l = str(gt).lower()
        label_l = str(label).lower()
        return (label_l == gt_l) or label_l.startswith(gt_l + ".")
    if dataset == 'camelyon16':
        def to_binary(x):
            if isinstance(x, (int, np.integer)):
                return int(x)

            x = str(x).strip().lower()
            if x in {"0", "normal", "negative"}:
                return 0
            elif x in {"1", "tumor", "positive"}:
                return 1
            else:
                raise ValueError(f"Unexpected CAMELYON16 label: {x}")

        return to_binary(label) == to_binary(gt)

    if dataset == 'tcga-gbmlgg':
        TCGA_MAP = {"LGG": "1", "GBM": "0"}
        mapped_gt = TCGA_MAP.get(str(gt), str(gt))
        return str(label) == mapped_gt

    return str(label).lower() == str(gt).lower()

def mrr(ranked_list, gt, dataset):
    for idx, item in enumerate(ranked_list):
        if match(gt, item, dataset):
            return 1.0 / (idx + 1)
    return 0.0

def ndcg_at_k(ranked_list, gt, k, dataset):
    # compute DCG
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        rel = 1.0 if match(gt, item, dataset) else 0.0
        dcg += rel / np.log2(i + 2)

    num_rel = sum(1 for l in ranked_list if match(gt, l, dataset))
    num_rel = min(num_rel, k)

    idcg = 0.0
    for i in range(num_rel):  
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--text_root", required=True)
    p.add_argument("--image_embs", required=True)
    p.add_argument("--slide_ids", required=True)
    p.add_argument("--csv_path", required=False)
    p.add_argument("--save_csv", default="top5_results.csv")
    return p.parse_args()

def main():
    args = parse_args()
    dataset = args.dataset.lower()

    text_dir = Path(args.text_root) / f"{dataset.upper()}_distractor"
    text_classes_all = np.load(text_dir / "text_class.npy", allow_pickle=True)
    selected_idx = get_positive_indices(text_classes_all, n_per_class=4)
    print(f"[INFO] Dataset: {dataset}, using text indices: {selected_idx}")

    text_dir = Path(args.text_root) / f"{dataset.upper()}_distractor"
    text_embs = np.load(text_dir / "text_emb.npy")
    text_classes = np.load(text_dir / "text_class.npy", allow_pickle=True)
    text_desc = np.load(text_dir / "text_desc.npy", allow_pickle=True)

    filtered_text_embs = text_embs[selected_idx]
    filtered_text_classes = text_classes[selected_idx]
    filtered_text_desc = text_desc[selected_idx]

    if dataset == 'camelyon17':
        filtered_text_classes = np.char.lower(filtered_text_classes)

    image_embs = np.load(args.image_embs)
    slide_ids = np.load(args.slide_ids, allow_pickle=True)

    if dataset == "tcga-gbmlgg":
        case_ids = [extract_tcga_case(str(s)) for s in slide_ids]
        df = pd.read_csv(args.csv_path)
        id_to_label = dict(zip(df["casename"], df["idh mutation"]))

        valid_embs, valid_slides, valid_labels = [], [], []
        for emb, cid, sid in zip(image_embs, case_ids, slide_ids):
            if cid in id_to_label:
                valid_embs.append(emb)
                valid_slides.append(sid)
                valid_labels.append(str(id_to_label[cid]))

        valid_embs = np.array(valid_embs)
        valid_labels = np.char.lower(np.array(valid_labels))
        print("[TCGA] matched images:", len(valid_embs))

    elif dataset == "unitopatho":
        valid_embs = image_embs
        valid_slides = slide_ids
        valid_labels = [extract_unitopatho_label(str(s)) for s in slide_ids]
        valid_labels = np.char.lower(np.array(valid_labels))
        print("[UNITO] parsed labels:", valid_labels[:10])

    elif dataset == "panda":
        panda_ids = [extract_panda_id(str(s)) for s in slide_ids]
        df = pd.read_csv(args.csv_path)

        id_to_label = dict(zip(df["image_id"], df["isup_grade"]))

        valid_embs, valid_slides, valid_labels = [], [], []
        for emb, pid, sid in zip(image_embs, panda_ids, slide_ids):
            if pid in id_to_label:
                valid_embs.append(emb)
                valid_slides.append(sid)
                valid_labels.append(str(id_to_label[pid]))

        valid_embs = np.array(valid_embs)
        valid_labels = np.char.lower(np.array(valid_labels))
        print("[PANDA] matched images:", len(valid_embs))

    elif dataset == "camelyon17":
        cam17_ids = [extract_cam17_id(str(s)) for s in slide_ids]
        df = pd.read_csv(args.csv_path)

        id_to_label = dict(zip(df["patient"], df["stage"]))

        valid_embs, valid_slides, valid_labels = [], [], []
        for emb, pid, sid in zip(image_embs, cam17_ids, slide_ids):
            if pid in id_to_label:
                valid_embs.append(emb)
                valid_slides.append(sid)
                valid_labels.append(str(id_to_label[pid]))

        valid_embs = np.array(valid_embs)
        valid_labels = np.char.lower(np.array(valid_labels))
        print("[CAM17] matched:", len(valid_embs))

    elif dataset == 'camelyon16':
        if args.csv_path is None:
            raise ValueError("CAMELYON16 requires --csv_path")

        df = pd.read_csv(args.csv_path)

        def normalize_id(x):
            x = str(x)
            if not x.endswith(".tif"):
                x = x + ".tif"
            return x

        slide_ids_norm = [normalize_id(s) for s in slide_ids]
        df["image"] = df["image"].astype(str)

        # ===== 2. 构建 image -> label 映射 =====
        label_map = dict(zip(df["image"], df["type"]))

        valid_embs = []
        valid_slides = []
        valid_labels = []

        for emb, sid in zip(image_embs, slide_ids_norm):
            if sid not in label_map:
                continue

            label_str = label_map[sid].lower()

            if label_str == "normal":
                label = 0
            elif label_str == "tumor":
                label = 1
            else:
                continue

            valid_embs.append(emb)
            valid_slides.append(sid)
            valid_labels.append(label)

        if len(valid_embs) == 0:
            raise ValueError("❌ No valid samples found after matching slide_ids with CSV")

        valid_embs = np.stack(valid_embs)
        valid_labels = np.array(valid_labels)

        print("\n[CAM16] loaded labels from CSV:", args.csv_path)
        print("Total samples:", len(valid_labels))
        print("Label distribution:\n", pd.Series(valid_labels).value_counts())

    results = []
    metrics_recall1 = []
    metrics_recall5 = []
    metrics_mrr = []
    metrics_ndcg5 = []

    for global_idx, t_emb, cls, desc in zip(
        selected_idx, filtered_text_embs, filtered_text_classes, filtered_text_desc
    ):

        sims = cosine_similarity(t_emb, valid_embs)
        top5 = sims.argsort()[::-1][:5]

        top5_slides = [valid_slides[j] for j in top5]
        top5_labels = [valid_labels[j] for j in top5]
        top5_scores = sims[top5]

        if dataset == "tcga-gbmlgg":
            TEXT_GT = {"LGG": "1", "GBM": "0"}   
            gt_label = TEXT_GT[cls]
        else:
            gt_label = str(cls)

        recall1 = int(match(gt_label, top5_labels[0], dataset))
        recall5 = int(any(match(gt_label, l, dataset) for l in top5_labels[:5]))
        rr = mrr(top5_labels, gt_label, dataset)
        ndcg = ndcg_at_k(top5_labels, gt_label, 5, dataset)

        metrics_recall1.append(recall1)
        metrics_recall5.append(recall5)
        metrics_mrr.append(rr)
        metrics_ndcg5.append(ndcg)

        results.append({
        "dataset": dataset,
        "text_index": global_idx,
        "text_class": cls,
        "text_description": desc,

        "top1_slide": top5_slides[0],
        "top1_label": top5_labels[0],
        "top1_score": float(top5_scores[0]),

        "top5_slides": "|".join(map(str, top5_slides)),
        "top5_labels": "|".join(map(str, top5_labels)),
        "top5_scores": "|".join([f"{s:.4f}" for s in top5_scores]),

        "recall@1": recall1,
        "recall@5": recall5,
        "mrr": rr,
        "ndcg@5": ndcg
    })

    print("\n===== RETRIEVAL METRICS =====")
    print(f"Recall@1: {np.mean(metrics_recall1):.4f}")
    print(f"Recall@5: {np.mean(metrics_recall5):.4f}")
    print(f"MRR:      {np.mean(metrics_mrr):.4f}")
    print(f"NDCG@5:   {np.mean(metrics_ndcg5):.4f}")

    pd.DataFrame({
        "Recall@1": [np.mean(metrics_recall1)],
        "Recall@5": [np.mean(metrics_recall5)],
        "MRR": [np.mean(metrics_mrr)],
        "NDCG@5": [np.mean(metrics_ndcg5)],
    }).to_csv("retrieval_metrics.csv", index=False)

    summary_row = {
        "dataset": dataset,
        "text_index": "OVERALL",
        "text_class": "OVERALL",
        "text_description": "SUMMARY",

        "top1_slide": "",
        "top1_label": "",
        "top1_score": "",

        "top5_slides": "",
        "top5_labels": "",
        "top5_scores": "",

        "recall@1": np.mean(metrics_recall1),
        "recall@5": np.mean(metrics_recall5),
        "mrr": np.mean(metrics_mrr),
        "ndcg@5": np.mean(metrics_ndcg5),
    }

    print("Saved retrieval_metrics.csv")

    df_results = pd.DataFrame(results)
    df_summary = pd.DataFrame([summary_row])

    final_df = pd.concat([df_results, df_summary], ignore_index=True)
    final_df.to_csv(args.save_csv, index=False)

    print(f"Saved ALL results (with summary) to {args.save_csv}")

    print("Filtered text_classes:", filtered_text_classes)
    print("Used text indices:", selected_idx)
    print("GT labels (first 20):", valid_labels[:20])
    print("Unique GT:", np.unique(valid_labels))


if __name__ == "__main__":
    main()