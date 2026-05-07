
import json
import torch
import numpy as np
from pathlib import Path
from load_model_UPDATE import load_fm
import argparse


def generate_text_embeddings(model_name, prompts_path, output_dir, device="cuda:1", ckpt_path=None, text_encoder=None):

    print(f"\n{'='*60}")
    print(f"Generating Text Embeddings with {model_name}")
    print(f"{'='*60}\n")
    
    # 1. 加载模型
    print(f"[1/4] Loading model: {model_name}...")
    if model_name == 'mi-zero':
        fm = load_fm(model_name, device=device, text_encoder=text_encoder or 'pubmedbert', ckpt_path=ckpt_path)
    else:
        fm = load_fm(model_name, device=device)
    
    if not fm.has_text_encoder:
        print(f"Error: {model_name} does not support text encoding!")
        print(f"Please use a model with text encoder (keep or plip)")
        return
    
    print(f"Model loaded")
    print(f"- Has text encoder: {fm.has_text_encoder}")
    
    print(f"\n[2/4] Loading prompts from: {prompts_path}")
    prompts = json.load(open(prompts_path))
    print(f"Loaded {len(prompts)} datasets")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/4] Output directory: {output_path}")

    print(f"\n[4/4] Encoding text prompts...")
    
    for dataset, mapping in prompts.items():
        labels = list(mapping.keys())
        texts = [
            mapping[k] if isinstance(mapping[k], str)
            else mapping[k][0]  
            for k in labels
        ]
        
        print(f"\n➡ Encoding {dataset} ({len(labels)} labels)")

        with torch.no_grad():
            text_emb = fm.encode_text(texts)
            text_emb = text_emb.cpu()
        
        emb_np = text_emb.numpy()

        npy_path = output_path / f"{dataset}.npy"
        np.save(npy_path, emb_np)
        print(f"   ✔ Saved NPY:  {npy_path}")

        json_path = output_path / f"{dataset}_labels.json"
        with open(json_path, 'w') as f:
            json.dump(labels, f, indent=4)
        print(f"Saved labels JSON: {json_path}")
        
        print(f"Shape = {tuple(emb_np.shape)}")
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  - Model: {model_name}")
    print(f"  - Datasets processed: {len(prompts)}")
    print(f"  - Output directory: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text embeddings from prompts using KEEP or PLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model name: keep or plip (models with text encoder)"
    )
    parser.add_argument(
        "-p", "--prompts",
        type=str,
        help="Path to prompts.json file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./text_embeddings",
        help="Output directory (default: ./text_embeddings)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device: cuda:0 or cpu (default: auto-detect)"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None, 
        help="Path to MI-Zero checkpoint (epoch_50.pt)"
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        choices=["bioclinicalbert", "pubmedbert"],
        help="Text encoder to use (default: pubmedbert)"
    )
    
    args = parser.parse_args()

    if not Path(args.prompts).exists():
        print(f"Error: Prompts file not found: {args.prompts}")
        return

    try:
        generate_text_embeddings(
            model_name=args.model,
            prompts_path=args.prompts,
            output_dir=args.output,
            device=args.device,
            ckpt_path=args.ckpt_path,
            text_encoder=args.text_encoder,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
