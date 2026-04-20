# TRIDENT VLM Feature Extraction Pipeline

This repository extends [Trident](https://github.com/mahmoodlab/TRIDENT) to support additional pathology vision-language foundation models (VLMs) for whole-slide image (WSI) feature extraction.

## Supported Models

| Model | Script | patch_size | Notes |
|---|---|---|---|
| `conch_v1` | `run_batch_of_slides.py` | 256 | Requires HF login + access |
| `musk` | `run_batch_of_slides.py` | 256 | Requires HF login + access |
| `plip` | `run_with_custom_fm.py` | 256 | Public |
| `keep` | `run_with_custom_fm.py` | 256 | Public |
| `pathgen-clip` | `run_with_custom_fm.py` | 256 | Requires HF login + access |
| `biomedclip-v2` | `run_with_custom_fm.py` | 256 | Public |
| `patho-clip` | `run_with_custom_fm.py` | 256 | Requires HF login + access |
| `mi-zero` | `run_with_custom_fm.py` | 256 (should mention in the paper) | Requires manual checkpoint download |
| `quiltnet` | `run_with_custom_fm.py` | 256 | Public |

## Installation

**1. Clone this repo**
```bash
git clone https://github.com/your_username/your_repo.git
cd your_repo
```

**2. Install Trident**
```bash
pip install -e .
```

**3. Install dependencies**
```bash
pip install torch==2.0.1
pip install torchvision==0.15.2
pip install transformers==4.44.0
pip install open_clip_torch
pip install timm==0.9.16
pip install numpy==1.24.0
```

## HuggingFace Setup

Some models require HuggingFace login and access approval. Run the following before using gated models:
```bash
huggingface-cli login
```

Then request access to the following models on HuggingFace:
- [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)
- [xiangjx/musk](https://huggingface.co/xiangjx/musk)
- [jamessyx/PathGen-CLIP](https://huggingface.co/jamessyx/PathGen-CLIP)
- [WenchuanZhang/Patho-CLIP-L](https://huggingface.co/WenchuanZhang/Patho-CLIP-L)

## MI-Zero Checkpoint

MI-Zero does not have a HuggingFace page. Download the checkpoint manually:

1. Visit https://github.com/mahmoodlab/MI-Zero
2. Find the Google Drive link in the README
3. Download one of the following checkpoints:
   - BioClinicalBERT: `ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt`
   - PubMedBERT: `ctranspath_448_pubmedbert/checkpoints/epoch_50.pt`
4. Place it anywhere on your server and pass the path via `--ckpt_path`

## Usage

### Step 1: Run the first model with `--task all` (generates seg + coords + feat)
```bash
python run_batch_of_slides.py --task all \
    --patch_encoder conch_v1 \
    --wsi_dir /path/to/wsis \
    --job_dir /path/to/output/conch \
    --mag 20 --patch_size 256 --gpu 0
```

### Step 2: Run remaining models reusing seg and coords (task set to feat)

Use `--seg_dir` to reuse existing segmentation results and `--coords_dir` to reuse existing patch coordinates. This avoids redundant computation and saves storage space.
```bash
# musk
python run_batch_of_slides.py --task feat \
    --patch_encoder musk \
    --wsi_dir /path/to/wsis \
    --job_dir /path/to/output/musk \
    --seg_dir /path/to/output/conch \
    --coords_dir /path/to/output/conch/20.0x_256px_0px_overlap \
    --mag 20 --patch_size 256 --gpu 0

# custom encoders
for model in plip keep pathgen-clip biomedclip-v2 patho-clip mstar quiltnet mi-zero; do
    python run_with_custom_fm.py --task all \
        --model $model \
        --wsi_dir /path/to/wsis \
        --job_dir /path/to/output/$model \
        --seg_dir /path/to/output/conch \
        --coords_dir /path/to/output/conch/20.0x_256px_0px_overlap \
        --mag 20 --patch_size 256 --gpu 0
done
```

> ⚠️ When using `--seg_dir` and `--coords_dir`, `--mag` and `--patch_size` must match the first model exactly, otherwise coordinates will not align correctly.
