import json
import torch
import numpy as np
from pathlib import Path
from load_model_UPDATE import load_fm
import argparse
from typing import List


@torch.inference_mode()
def encode_texts(fm, texts: List[str]) -> np.ndarray:
    text_emb = fm.encode_text(texts)
    return text_emb.cpu().numpy()


def save_output(dir_path: Path, emb: np.ndarray, descs: List[str], classes: List[str]):
    dir_path.mkdir(exist_ok=True, parents=True)
    np.save(dir_path / "text_emb.npy", emb)
    np.save(dir_path / "text_desc.npy", np.array(descs))
    np.save(dir_path / "text_class.npy", np.array(classes))
    print(f"💾 Saved → {dir_path} | Shape: {emb.shape}, Descriptions: {len(descs)}")


def main():
    parser = argparse.ArgumentParser(description="Generate text embeddings for each dataset")
    parser.add_argument("--model", "-m", required=True, help="Model name: keep, plip, pathgen-clip 等")
    parser.add_argument("--json", required=True, help="distractor JSON path")
    parser.add_argument("--output", "-o", required=True, help="output")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path", default=None, help="mi-zero required only")
    parser.add_argument("--text_encoder", default=None, choices=["bioclinicalbert", "pubmedbert"])
    args = parser.parse_args()

    if not Path(args.json).exists():
        print(f"JSON file not found: {args.json}")
        return

    print(f"[1/3] Loading model: {args.model}...")
    if args.model == "mi-zero":
        fm = load_fm(args.model, device=args.device, ckpt_path=args.ckpt_path,
                     text_encoder=args.text_encoder or "pubmedbert")
    else:
        fm = load_fm(args.model, device=args.device)

    if not fm.has_text_encoder:
        print(f"{args.model} does not support text encoding!")
        return
    print(f"Model loaded: {fm.name}")

    print(f"[2/3] Loading JSON...")
    with open(args.json) as f:
        data = json.load(f)

    print(f"[3/3] Encoding...")
    out_root = Path(args.output)
    out_root.mkdir(exist_ok=True, parents=True)

    for dataset, class_dict in data.items():
        descs, classes = [], []
        for cls_name, content in class_dict.items():
            if isinstance(content, dict):
                for desc in content.get("positives", []) + content.get("negatives", []):
                    descs.append(desc)
                    classes.append(cls_name)

        print(f"{dataset}: {len(descs)} descriptions")
        emb = encode_texts(fm, descs)
        save_output(out_root / f"{dataset}_distractor", emb, descs, classes)

    print("DONE.")


if __name__ == "__main__":
    main()