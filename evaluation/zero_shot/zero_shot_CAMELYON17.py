import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
import argparse
import pandas as pd
from pathlib import Path

def zero_shot_classification(image_embs, text_embs, labels, normalize=True, verbose=True):
    if isinstance(image_embs, np.ndarray):
        image_embs = torch.tensor(image_embs, dtype=torch.float32)
    if isinstance(text_embs, np.ndarray):
        text_embs = torch.tensor(text_embs, dtype=torch.float32)

    if normalize:
        image_embs = F.normalize(image_embs, dim=-1)
        text_embs = F.normalize(text_embs, dim=-1)

    sims = image_embs @ text_embs.T
    preds = sims.argmax(dim=1).cpu().numpy()
    labels = np.array(labels)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    sims_np = sims.cpu().numpy()
    labels_bin = label_binarize(labels, classes=np.arange(text_embs.shape[0]))

    try:
        auc = roc_auc_score(labels_bin, sims_np, average="macro", multi_class="ovr")
    except:
        auc = None

    if verbose:
        print("\n=========== ZERO-SHOT RESULTS ===========")
        print(f"Accuracy:     {acc:.4f}")
        print(f"Macro F1:     {f1:.4f}")
        print(f"Macro AUC:    {auc if auc is None else round(auc,4)}")
        print("==========================================\n")

    return preds, sims_np, acc, f1, auc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_emb", required=True)
    parser.add_argument("--slide_ids", required=True)
    parser.add_argument("--text_emb", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save results")
    return parser.parse_args()

def main():
    args = parse_args()

    image_embs = np.load(args.image_emb)
    slide_ids = np.load(args.slide_ids)

    print(f"Loaded image_embs: {image_embs.shape}")
    print(f"Loaded slide_ids:  {len(slide_ids)}")

    df = pd.read_csv(args.csv_path)

    id_to_label = dict(zip(df["patient"], df["stage"]))

    LABEL_MAP = {
        "negative": 0,
        "itc": 1,
        "micro": 2,
        "macro": 3,
    }

    valid_labels = []
    valid_slide_ids = []
    missing_slides = []
    unknown_labels = []

    for sid in slide_ids:
        key = f"{sid}.tif"

        if key not in id_to_label:
            missing_slides.append(sid)
            continue

        stage = id_to_label[key].lower()

        if stage not in LABEL_MAP:
            unknown_labels.append((sid, stage))
            continue

        valid_labels.append(LABEL_MAP[stage])
        valid_slide_ids.append(sid)

    valid_labels = np.array(valid_labels, dtype=int)

    mask = np.array([sid in valid_slide_ids for sid in slide_ids])
    image_embs = image_embs[mask]

    text_embs = np.load(args.text_emb)

    print(f"\nMatched {len(valid_slide_ids)} / {len(slide_ids)} slides.")
    print(f"Image embeddings after filtering: {image_embs.shape}")
    print(f"Text embeddings shape: {text_embs.shape}")

    if missing_slides:
        print("Missing slides (not in metadata):")
        for x in missing_slides[:10]:
            print("   ", x)
        if len(missing_slides) > 10:
            print(f"   ... and {len(missing_slides)-10} more")

    if unknown_labels:
        print("Unknown stage labels encountered:")
        for sid, st in unknown_labels:
            print(f"   {sid} → {st}")

    preds, sims, acc, f1, auc = zero_shot_classification(
        image_embs=image_embs,
        text_embs=text_embs,
        labels=valid_labels,
        verbose=args.verbose,
    )

    print("🎉 Finished CAMELYON17 Zero-shot Evaluation.\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
    "accuracy": float(acc),
    "f1_macro": float(f1),
    "auc": None if auc is None else float(auc)
    }

    import json
    with open(output_dir / f"results_zeroshot.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()