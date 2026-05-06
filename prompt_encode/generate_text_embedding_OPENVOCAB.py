from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List

from load_model_UPDATE import load_fm

SETS = [
    ("set1_openvocab",   "open_vocab_set1.json"),
    ("set2_finegrained", "open_vocab_set2_finegrained.json"),
    ("set3_crossorgan",  "open_vocab_set3_crossorgan.json"),
    ("set4_alldatasets", "open_vocab_set4_alldatasets.json"),
]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def encode(fm, texts: List[str]) -> np.ndarray:
    with torch.no_grad():
        emb = fm.encode_text(texts)
    return emb.cpu().numpy()


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"{path.name}  shape={arr.shape}")


def gen_set(fm, set_name: str, dataset: str, data: dict, out_dir: Path) -> None:
    label_data = data[dataset]
    labels = list(label_data.keys())

    for lbl in labels:
        cands = label_data[lbl]["candidates"]
        print(f"      [{lbl}] {len(cands)} candidates")

        emb = normalize(encode(fm, cands))  
        save(emb, out_dir / f"{dataset}_{lbl}_candidates.npy")
        (out_dir / f"{dataset}_{lbl}_candidate_texts.json").write_text(
            json.dumps(cands, indent=2)
        )

    ground_truths = {lbl: label_data[lbl]["ground_truth"] for lbl in labels}
    (out_dir / f"{dataset}_ground_truth.json").write_text(
        json.dumps(ground_truths, indent=2)
    )
    (out_dir / f"{dataset}_labels.json").write_text(
        json.dumps(labels, indent=2)
    )
    print(f"metadata saved")

def main():
    parser = argparse.ArgumentParser(description="Generate open vocabulary text embeddings per label")
    parser.add_argument("-m", "--model",    required=True,  help="Model name: plip, conch, keep, ...")
    parser.add_argument("--prompt_dir",     required=True,  help="Directory containing open_vocab_set*.json files")
    parser.add_argument("-o", "--output",   required=True,  help="Output root directory")
    parser.add_argument("--datasets",       nargs="+",      default=None, help="Subset of datasets (default: all in JSON)")
    parser.add_argument("--sets",           nargs="+",      default=None, help="Subset of sets e.g. set1_openvocab (default: all)")
    parser.add_argument("--device",         default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path",      default=None,   help="Checkpoint path (required for mi-zero)")
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    out_root   = Path(args.output)

    sets_to_run = [(name, fname) for name, fname in SETS
                   if args.sets is None or name in args.sets]

    print(f"\n{'='*60}")
    print(f"  Model  : {args.model}")
    print(f"  Sets   : {[s for s, _ in sets_to_run]}")
    print(f"  Output : {out_root}")
    print(f"{'='*60}\n")

    kwargs = {}
    if args.ckpt_path:
        kwargs["ckpt_path"] = args.ckpt_path

    fm = load_fm(args.model, device=args.device, **kwargs)
    assert fm.has_text_encoder, f"{args.model} has no text encoder!"
    print(f"Model loaded\n")

    for set_name, fname in sets_to_run:
        json_path = prompt_dir / fname
        if not json_path.exists():
            print(f"Skipping {set_name}: {json_path} not found")
            continue

        data     = load_json(json_path)
        datasets = args.datasets or list(data.keys())
        out_dir  = out_root / args.model / set_name

        print(f"\n── {set_name} ──")
        for dataset in datasets:
            if dataset not in data:
                print(f"   ⚠ {dataset} not in {fname}, skipping")
                continue
            print(f"   {dataset}")
            gen_set(fm, set_name, dataset, data, out_dir)

    print(f"All done → {out_root}")


if __name__ == "__main__":
    main()