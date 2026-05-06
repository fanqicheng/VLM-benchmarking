from __future__ import annotations
import krippendorff
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# ---------------------------------------------------------------------------
# Pool axis 1 layout
# ---------------------------------------------------------------------------
V1_INDICES = list(range(0, 5))
V2_INDICES = {"short": 5, "medium": 6, "long": 7, "clinical": 8}
V2_LEVELS  = ["short", "medium", "long", "clinical"]
ALL_INDICES = list(range(9))

N_ENSEMBLE_SAMPLES = 100
SEED = 42

LABEL_MAPS = {
    "CAMELYON16": {"normal": 0, "tumor": 1},
    "CAMELYON17": {"negative": 0, "itc": 1, "micro": 2, "macro": 3},
    "UNITOPATHO": {"norm": 0, "hp": 1, "ta.lg": 2, "ta.hg": 3, "tva.lg": 4, "tva.hg": 5},
    "TCGA-GBMLGG": {"lgg": 0, "gbm": 1},
    "PANDA": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
}

def extract_unitopatho_label(slide_name):
    """
    Supports:
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


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def predict(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """
    image_emb : [N_slides, D]
    text_emb  : [N_classes, D]
    returns   : [N_slides] predicted class indices
    """
    sims = image_emb @ text_emb.T     
    return sims.argmax(axis=1)


def prediction_consistency(all_preds: np.ndarray) -> float:
    """
    all_preds : [N_variants, N_slides]
    Returns % of slides where ALL variants agree on the same prediction.
    """
    agree = np.all(all_preds == all_preds[0], axis=0)
    return float(np.mean(agree))


def prompt_stability_score(all_preds: np.ndarray) -> float:
    try:
        return float(krippendorff.alpha(reliability_data=all_preds, level_of_measurement="nominal"))
    except ValueError:
        return None  


def align_labels(slide_ids: np.ndarray, csv_path: Path, dataset: str):
    label_map = LABEL_MAPS[dataset]

    if dataset in ("UNITOPATHO",):
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
        id_to_label = dict(zip(
            df["image_id"].astype(str),
            df["isup_grade"].astype(str)
        ))
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


def evaluate_variant1(image_emb: np.ndarray, pool: np.ndarray, gt: np.ndarray) -> dict:
    """
    Variant 1: 5 paraphrase variants, one prediction each.
    pool[:, 0..4, :]
    """
    accs, balanced_accs, all_preds = [], [], [] 

    for vi in V1_INDICES:
        text_emb = pool[:, vi, :]             
        preds    = predict(image_emb, text_emb)
        accs.append(accuracy_score(gt, preds))
        balanced_accs.append(balanced_accuracy_score(gt, preds))  
        all_preds.append(preds)

    all_preds = np.stack(all_preds, axis=0)   
    pss = prompt_stability_score(all_preds)


    return {
        "accuracies":             {f"v{i}": round(accs[i], 4) for i in range(5)},
        "mean_accuracy":          round(float(np.mean(accs)), 4),
        "accuracy_std":           round(float(np.std(accs)), 4),
        "prediction_consistency": round(prediction_consistency(all_preds), 4),
        "prompt_stability_score": round(pss, 4) if pss is not None else None,
        "mean_balanced_accuracy": round(float(np.mean(balanced_accs)), 4),
    }


def evaluate_variant2(image_emb: np.ndarray, pool: np.ndarray, gt: np.ndarray) -> dict:
    """
    Variant 2: 4 length levels, one prediction each.
    pool[:, 5..8, :]
    """
    accs, balanced_accs, all_preds = [], [], [] 

    for level in V2_LEVELS:
        text_emb = pool[:, V2_INDICES[level], :]  
        preds    = predict(image_emb, text_emb)
        accs.append(accuracy_score(gt, preds))
        balanced_accs.append(balanced_accuracy_score(gt, preds))  
        all_preds.append(preds)

    all_preds = np.stack(all_preds, axis=0)        
    pss = prompt_stability_score(all_preds)


    return {
        "accuracies":             {lvl: round(accs[i], 4) for i, lvl in enumerate(V2_LEVELS)},
        "mean_accuracy":          round(float(np.mean(accs)), 4),
        "accuracy_std":           round(float(np.std(accs)), 4),
        "prediction_consistency": round(prediction_consistency(all_preds), 4),
        "prompt_stability_score": round(pss, 4) if pss is not None else None,
        "mean_balanced_accuracy": round(float(np.mean(balanced_accs)), 4),
    }


def evaluate_variant3(image_emb: np.ndarray, pool: np.ndarray, gt: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    results = {}

    for n in range(1, 10):
        if n == 9:
            text_emb = pool.mean(axis=1)
            text_emb = normalize(text_emb)
            preds    = predict(image_emb, text_emb)
            acc      = accuracy_score(gt, preds)
            bacc     = balanced_accuracy_score(gt, preds)
            results[n] = {
                "mean_accuracy":          round(acc, 4),
                "mean_balanced_accuracy": round(bacc, 4),
                "std":                    0.0,
                "balanced_std":           0.0,
                "ci95_lower":             round(acc, 4),
                "ci95_upper":             round(acc, 4),
                "iqr":                    0.0,
                "balanced_ci95_lower":    round(bacc, 4),
                "balanced_ci95_upper":    round(bacc, 4),
                "balanced_iqr":           0.0,
            }
        else:
            sample_accs, sample_baccs = [], []
            for _ in range(N_ENSEMBLE_SAMPLES):
                indices  = rng.choice(ALL_INDICES, size=n, replace=False)
                text_emb = pool[:, indices, :].mean(axis=1)
                text_emb = normalize(text_emb)
                preds    = predict(image_emb, text_emb)
                sample_accs.append(accuracy_score(gt, preds))
                sample_baccs.append(balanced_accuracy_score(gt, preds))

            sample_accs  = np.array(sample_accs)
            sample_baccs = np.array(sample_baccs)

            ci95_lower, ci95_upper         = np.percentile(sample_accs,  [2.5, 97.5])
            bci95_lower, bci95_upper       = np.percentile(sample_baccs, [2.5, 97.5])
            iqr  = float(np.percentile(sample_accs,  75) - np.percentile(sample_accs,  25))
            biqr = float(np.percentile(sample_baccs, 75) - np.percentile(sample_baccs, 25))

            results[n] = {
                "mean_accuracy":          round(float(np.mean(sample_accs)), 4),
                "mean_balanced_accuracy": round(float(np.mean(sample_baccs)), 4),
                "std":                    round(float(np.std(sample_accs)), 4),
                "balanced_std":           round(float(np.std(sample_baccs)), 4),
                "ci95_lower":             round(float(ci95_lower), 4),
                "ci95_upper":             round(float(ci95_upper), 4),
                "iqr":                    round(iqr, 4),
                "balanced_ci95_lower":    round(float(bci95_lower), 4),
                "balanced_ci95_upper":    round(float(bci95_upper), 4),
                "balanced_iqr":           round(biqr, 4),
            }

        print(f"   N={n:2d}  mean_acc={results[n]['mean_accuracy']:.4f}"
              f"  bacc={results[n]['mean_balanced_accuracy']:.4f}"
              f"  std={results[n]['std']:.4f}"
              f"  CI95=[{results[n]['ci95_lower']:.4f}, {results[n]['ci95_upper']:.4f}]"
              f"  IQR={results[n]['iqr']:.4f}")

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Paraphrase robustness evaluation")
    parser.add_argument("--image_emb",  required=True, help="Path to image embeddings .npy")
    parser.add_argument("--slide_ids",  required=True, help="Path to slide ids .npy")
    parser.add_argument("--pool_emb",   required=True, help="Path to pool embeddings .npy [N_classes, 9, D]")
    parser.add_argument("--csv_path",   required=False, help="Path to ground truth CSV")
    parser.add_argument("--dataset",    required=True, help="Dataset name (e.g. CAM16)")
    parser.add_argument("--model",      required=True, help="Model name (e.g. plip)")
    parser.add_argument("--output_dir", default="./results/paraphrase_robustness")
    return parser.parse_args()


def main():
    args = parse_args()

    image_emb = normalize(np.load(args.image_emb)) 
    slide_ids = np.load(args.slide_ids)
    pool      = np.load(args.pool_emb)               

    pool = normalize(pool.reshape(-1, pool.shape[-1])).reshape(pool.shape)

    gt, valid_indices = align_labels(slide_ids, Path(args.csv_path) if args.csv_path else None, args.dataset)
    image_emb = image_emb[valid_indices]
    slide_ids = slide_ids[valid_indices]

    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Model   : {args.model}")
    print(f"  Slides  : {len(gt)}")
    print(f"  Pool    : {pool.shape}")
    print(f"{'='*60}\n")

    print("[Variant 1] Paraphrase variants...")
    v1_results = evaluate_variant1(image_emb, pool, gt)
    print(f"  mean_acc={v1_results['mean_accuracy']}  std={v1_results['accuracy_std']}  PSS={v1_results['prompt_stability_score']}")

    print("\n[Variant 2] Prompt length...")
    v2_results = evaluate_variant2(image_emb, pool, gt)
    print(f"  mean_acc={v2_results['mean_accuracy']}  std={v2_results['accuracy_std']}  PSS={v2_results['prompt_stability_score']}")

    print("\n[Variant 3] Ensemble gain...")
    v3_results = evaluate_variant3(image_emb, pool, gt)

    output = {
        "dataset":   args.dataset,
        "model":     args.model,
        "variant1":  v1_results,
        "variant2":  v2_results,
        "variant3":  v3_results,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.model}.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()