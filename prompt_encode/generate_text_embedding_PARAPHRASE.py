from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List

from load_model_UPDATE import load_fm

LENGTH_LEVELS = ["short", "medium", "long", "clinical"]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_labels(data: dict, dataset: str) -> List[str]:
    return list(data[dataset].keys())


def encode(fm, texts: List[str]) -> np.ndarray:
    with torch.no_grad():
        emb = fm.encode_text(texts)
    return emb.cpu().numpy()


def save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(f"   ✔ {path.name}  shape={arr.shape}")


def gen_pool(fm, dataset: str, model: str, v1_data: dict, v2_data: dict, out_dir: Path) -> None:
    """
    Per class label, pool = V1 (5 paraphrases) + V2 (4 length levels) = 9 prompts
    axis 1 layout:
        [0]  V1 paraphrase 0
        [1]  V1 paraphrase 1
        [2]  V1 paraphrase 2
        [3]  V1 paraphrase 3
        [4]  V1 paraphrase 4
        [5]  V2 short
        [6]  V2 medium
        [7]  V2 long
        [8]  V2 clinical
    """
    labels = get_labels(v1_data, dataset)
    all_embs = []

    for lbl in labels:
        pool_texts = (
            v1_data[dataset][lbl]                                     
            + [v2_data[dataset][lbl][lvl] for lvl in LENGTH_LEVELS]  
        )  
        assert len(pool_texts) == 9, f"Expected 9 prompts for {lbl}, got {len(pool_texts)}"

        emb = encode(fm, pool_texts)   
        all_embs.append(emb)

    stacked = np.stack(all_embs, axis=0)   

    save(stacked, out_dir / f"{dataset}_pool_{model}.npy")

    labels_path = out_dir / f"{dataset}_labels_{model}.json"
    labels_path.write_text(json.dumps(labels, indent=2))
    print(f"{labels_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate paraphrase robustness text embeddings")
    parser.add_argument("-m", "--model",   required=True,
                        help="Model name: plip, conch, keep, biomedclip-v2, ...")
    parser.add_argument("--prompt_dir",    required=True,
                        help="Directory containing variant1.json and variant2.json")
    parser.add_argument("-o", "--output",  default="/Data3/shangke/paraphrase_text_emb",
                        help="Output root directory")
    parser.add_argument("--datasets",      nargs="+", default=None,
                        help="Subset of datasets (default: all in JSON)")
    parser.add_argument("--device",        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path", default=None, help="Checkpoint path (required for mi-zero)")
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    v1_data = load_json(prompt_dir / "variant1.json")
    v2_data = load_json(prompt_dir / "variant2.json")

    datasets = args.datasets or list(v1_data.keys())
    out_dir  = Path(args.output)

    print(f"\n{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  Datasets : {datasets}")
    print(f"  Output   : {out_dir}")
    print(f"  Pool layout (axis 1):")
    print(f"    [0..4]  V1 paraphrases")
    print(f"    [5..8]  V2 short / medium / long / clinical")
    print(f"{'='*60}\n")

    kwargs = {}
    if args.ckpt_path:
        kwargs["ckpt_path"] = args.ckpt_path
    fm = load_fm(args.model, device=args.device, **kwargs)
    assert fm.has_text_encoder, f"{args.model} has no text encoder!"
    print(f"Model loaded\n")

    for dataset in datasets:
        print(f"\n── {dataset} ──")
        gen_pool(fm, dataset, args.model, v1_data, v2_data, out_dir)

    print(f"All done → {out_dir}")


if __name__ == "__main__":
    main()