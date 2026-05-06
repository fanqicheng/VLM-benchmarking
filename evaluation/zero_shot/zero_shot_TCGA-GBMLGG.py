import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pandas as pd
import json


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

    unique_classes = sorted(np.unique(labels))
    C = len(unique_classes)

    sims_np = sims_np[:, :C]

    if C == 2:

        try:
            positive_scores = sims_np[:, 1]
            auc_score = roc_auc_score(labels, positive_scores)
        except Exception as e:
            print("Binary AUC error:", e)
            auc_score = None
    else:

        try:
            labels_bin = label_binarize(labels, classes=unique_classes)
            auc_score = roc_auc_score(
                labels_bin,
                sims_np,
                average="macro",
                multi_class="ovr",
            )
        except Exception as e:
            print("Macro AUC error:", e)
            auc_score = None

    if verbose:
        print("=========== ZERO-SHOT RESULTS ===========")
        print(f"Accuracy:     {acc:.4f}")
        print(f"Macro F1:     {f1:.4f}")
        if C == 2:
            print(f"Binary AUC:   {auc_score if auc_score is None else round(auc_score, 4)}")
        else:
            print(f"Macro AUC:    {auc_score if auc_score is None else round(auc_score, 4)}")
        print("==========================================")

    return preds, sims_np, acc, f1, auc_score

def plot_confusion_matrix(labels, preds, class_names, save_path=None):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(labels, sims, class_names, save_path=None):

    labels = np.array(labels)

    # ensure class_names sorted and numeric
    class_names = sorted(list(class_names))
    C = len(class_names)

    # proper binarization using class_names
    labels_bin = label_binarize(labels, classes=[0,1])
    if labels_bin.shape[1] == 1:
        labels_bin = np.hstack([1 - labels_bin, labels_bin])  

    plt.figure(figsize=(7, 6))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], sims[:, i])
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC={score:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_emb", required=True)
    parser.add_argument("--slide_ids", required=True)
    parser.add_argument("--text_emb", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_embs = np.load(args.image_emb)
    slide_ids = np.load(args.slide_ids)

    text_embs = np.load(args.text_emb, allow_pickle=True)
    text_embs = np.stack(text_embs, axis=0)  
    text_embs = text_embs[[1, 0], :] 

    df = pd.read_csv(args.csv_path)
    id_to_label = dict(zip(df["casename"], df["idh mutation"]))

    clean_ids = ["-".join(str(sid).split("-")[:3]) for sid in slide_ids]

    valid_idx = [i for i, cid in enumerate(clean_ids) if cid in id_to_label]
    filtered_image_embs = image_embs[valid_idx]
    filtered_clean_ids = [clean_ids[i] for i in valid_idx]

    print(f"\n➡ Total slides: {len(slide_ids)}")
    print(f"➡ Valid slides with label: {len(filtered_clean_ids)}")
    print(f"➡ Removed (no label): {len(slide_ids) - len(filtered_clean_ids)}\n")

    filtered_labels = np.array([id_to_label[cid] for cid in filtered_clean_ids], dtype=int)

    class_names = sorted(list(set(filtered_labels)))

    preds, sims, acc, f1, auc_score = zero_shot_classification(
        image_embs=filtered_image_embs,
        text_embs=text_embs,
        labels=filtered_labels,
        verbose=args.verbose,
    )

    print("DEBUG class_names:", class_names)
    print("DEBUG unique labels:", np.unique(filtered_labels))

    plot_confusion_matrix(filtered_labels, preds, class_names,
                          save_path=output_dir / "confusion_matrix.png")

    plot_roc_curve(filtered_labels, sims, class_names,
                   save_path=output_dir / "roc_curve.png")

    results = {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "auc": None if auc_score is None else float(auc_score)
    }
    with open(output_dir / "results_zeroshot.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()