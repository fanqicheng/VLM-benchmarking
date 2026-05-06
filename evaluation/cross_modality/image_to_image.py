import numpy as np
from pathlib import Path
import argparse
import pandas as pd
import re
import torch
import json
from evaluation_image_image import evaluate_ranking
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["TCGA-GBMLGG", "PANDA", "CAMELYON17", "CAMELYON16", "UNITOPATHO"],     ### NEW
                        help="Choose which dataset")

    parser.add_argument("--image_emb", required=True)
    parser.add_argument("--slide_ids", required=True)
    parser.add_argument("--csv_path", required=False, default=None)
    parser.add_argument("--out_json", required=True)
    return parser.parse_args()

def extract_tcga_case(text):
    m = re.match(r"(TCGA-\d{2}-\d{4})", text)
    return m.group(1) if m else None


def extract_panda_id(text):
    return str(text)


def extract_cam17_id(text):
    """Slide ids are like: patient_000_node_0  → match to patient_000_node_0.tif"""
    return f"{text}.tif"

def extract_cam16_id(text):
    return f"{text}.tif"

def main():
    args = parse_args()

    image_embs = np.load(args.image_emb)
    slide_ids = np.load(args.slide_ids)

    print("Loaded embeddings:", image_embs.shape)
    print("Loaded slide_ids:", slide_ids.shape)

    # Dataset-specific label logic

    if args.dataset == "UNITOPATHO":
        UNITO_LABELS = ["TVA.HG", "TVA.LG", "TA.HG", "TA.LG", "HP", "NORM"]
        def extract_unito(name):
            name = name.upper()
            for lbl in UNITO_LABELS:
                if lbl.replace(".", "") in name.replace(".", ""):
                    return lbl
            raise ValueError(f"Cannot parse: {name}")
        
        valid_embs = np.array(list(image_embs))
        valid_labels = np.array([extract_unito(str(s)) for s in slide_ids])
        print(f"Matched {len(valid_labels)} UNITOPATHO slides")

    else:
        df = pd.read_csv(args.csv_path)

        if args.dataset == "TCGA-GBMLGG":
            ids = [extract_tcga_case(str(s)) for s in slide_ids]
            key_column = "casename"
            label_column = "idh mutation"

        elif args.dataset == "PANDA":
            ids = [extract_panda_id(str(s)) for s in slide_ids]
            key_column = "image_id"
            label_column = "isup_grade"

        elif args.dataset == "CAMELYON17":
            ids = [extract_cam17_id(str(s)) for s in slide_ids]
            key_column = "patient"
            label_column = "stage"
            CAM17_MAP = {"negative": 0, "itc": 1, "micro": 2, "macro": 3}

        elif args.dataset == "CAMELYON16":
            ids = [extract_cam16_id(str(s)) for s in slide_ids]
            key_column = "image"
            label_column = "class"

        id_to_label = dict(zip(df[key_column], df[label_column]))
        valid_embs = []
        valid_labels = []

        for emb, sid in zip(image_embs, ids):
            if sid not in id_to_label:
                continue
            label = id_to_label[sid]
            if args.dataset == "CAMELYON17":
                label = label.lower()
                if label not in CAM17_MAP:
                    continue
                label = CAM17_MAP[label]
            valid_embs.append(emb)
            valid_labels.append(label)

        valid_embs = np.array(valid_embs)
        valid_labels = np.array(valid_labels)
        print(f"Matched {len(valid_labels)} images with metadata")

   
    if args.dataset != "CAMELYON17":
        class_names = sorted(set(valid_labels))
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        labels_num = np.array([class_to_idx[x] for x in valid_labels])
    else:
        labels_num = valid_labels  

    sims, ranking, metrics = evaluate_ranking(
        query_embs=valid_embs,
        db_embs=valid_embs,
        db_labels=labels_num,
        true_labels=labels_num
    )

    results = {
    "retrieval_metrics": {k: float(v) for k, v in metrics.items()},
    }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved result → {args.out_json}")


if __name__ == "__main__":
    main()