# PathLang: A Language-Centered Benchmark for Vision-Language Models in Computational Pathology

PathLang is a comprehensive benchmark for evaluating vision-language models (VLMs) on pathology whole-slide images (WSIs) across multiple language-centered tasks, including zero-shot classification, cross-modal retrieval, open vocabulary retrieval, and paraphrase robustness.

---

## Repository Structure

```
VLM-benchmarking/
  ├── preprocessing/       # WSI feature extraction pipeline
  ├── prompt_encode/       # Text prompt embedding generation
  └── evaluation/          # Retrieval and alignment evaluation
      ├── cross_modality/
      ├── open_vocab/
      ├── paraphrase_robustness/
      └── zero_shot/
```

---

## Supported Models

| Model | Script | patch_size | Notes |
|-------|--------|-----------|-------|
| conch_v1 | `run_batch_of_slides.py` | 256 | Requires HF login + access |
| musk | `run_batch_of_slides.py` | 256 | Requires HF login + access |
| plip | `run_with_custom_fm.py` | 256 | Public |
| keep | `run_with_custom_fm.py` | 256 | Public |
| pathgen-clip | `run_with_custom_fm.py` | 256 | Requires HF login + access |
| biomedclip-v2 | `run_with_custom_fm.py` | 256 | Public |
| patho-clip | `run_with_custom_fm.py` | 256 | Requires HF login + access |
| mi-zero | `run_with_custom_fm.py` | 256 | Requires manual checkpoint download |
| quiltnet | `run_with_custom_fm.py` | 256 | Public |

---

## Installation

**1. Clone this repo**
```bash
git clone https://github.com/[anonymous]/[anonymous].git
cd VLM-benchmarking
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

**4. HuggingFace Setup**

Some models require HuggingFace login and access approval:
```bash
huggingface-cli login
```

Then request access to the following models on HuggingFace:
- [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)
- [xiangjx/musk](https://huggingface.co/xiangjx/musk)
- [jamessyx/PathGen-CLIP](https://huggingface.co/jamessyx/PathGen-CLIP)
- [WenchuanZhang/Patho-CLIP-L](https://huggingface.co/WenchuanZhang/Patho-CLIP-L)

**5. MI-Zero Checkpoint**

MI-Zero does not have a HuggingFace page. Download the checkpoint manually:
- Visit https://github.com/mahmoodlab/MI-Zero
- Find the Google Drive link in the README
- Download one of the following checkpoints:
  - BioClinicalBERT: `ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt`
  - PubMedBERT: `ctranspath_448_pubmedbert/checkpoints/epoch_50.pt`
- Place it anywhere and pass the path via `--ckpt_path`

---

## Step 1: Preprocessing — WSI Feature Extraction

### PANDA, TCGA, CAMELYON16/17

**Run the first model with `--task all` (generates seg + coords + features):**
```bash
python preprocessing/run_batch_of_slides.py --task all \
    --patch_encoder conch_v1 \
    --wsi_dir /path/to/wsis \
    --job_dir /path/to/output/conch \
    --mag 20 --patch_size 256 --gpu 0
```

**Run remaining models reusing seg and coords:**
```bash
# musk
python preprocessing/run_batch_of_slides.py --task feat \
    --patch_encoder musk \
    --wsi_dir /path/to/wsis \
    --job_dir /path/to/output/musk \
    --seg_dir /path/to/output/conch \
    --coords_dir /path/to/output/conch/20.0x_256px_0px_overlap \
    --mag 20 --patch_size 256 --gpu 0

# custom encoders
for model in plip keep pathgen-clip biomedclip-v2 patho-clip quiltnet mi-zero; do
    python preprocessing/run_with_custom_fm.py --task feat \
        --model $model \
        --wsi_dir /path/to/wsis \
        --job_dir /path/to/output/$model \
        --seg_dir /path/to/output/conch \
        --coords_dir /path/to/output/conch/20.0x_256px_0px_overlap \
        --mag 20 --patch_size 256 --gpu 0
done
```

> ⚠️ When using `--seg_dir` and `--coords_dir`, `--mag` and `--patch_size` must match the first model exactly.

### UnitoPatho

```bash
# Step 1 — seg + coords + features (run once)
python preprocessing/run_unitopatho.py --task all \
    --patch_encoder conch_v1 \
    --wsi_dir /path/to/unitopatho/ \
    --job_dir /path/to/output/ \
    --mag 20 --patch_size 256 --gpu 0

# Step 2 — features only for remaining models
python preprocessing/run_unitopatho.py --task feat \
    --model plip \
    --wsi_dir /path/to/unitopatho/ \
    --job_dir /path/to/output/ \
    --mag 20 --patch_size 256 --gpu 0
```

### Aggregate Features

```bash
bash preprocessing/run_aggregate.sh --dataset CAM16 --top_k 0.05
```

---

## Step 2: Prompt Encoding — Text Embedding Generation

Generates text embeddings from expert-written clinical prompts for each evaluation task.

```bash
# Zero-shot classification prompts
python prompt_encode/generate_text_embedding_ZEROSHOT.py \
    --model plip --dataset CAMELYON16 \
    --output_dir /path/to/text_embeddings

# Cross-modal retrieval prompts
python prompt_encode/generate_text_embedding_CROSS_MODAL.py \
    --model plip --dataset CAMELYON16 \
    --output_dir /path/to/text_embeddings

# Open vocabulary prompts
python prompt_encode/generate_text_embedding_OPENVOCAB.py \
    --model plip --dataset CAMELYON16 \
    --output_dir /path/to/text_embeddings

# Paraphrase robustness prompts
python prompt_encode/generate_text_embedding_PARAPHRASE.py \
    --model plip --dataset CAMELYON16 \
    --output_dir /path/to/text_embeddings
```

---

## Step 3: Evaluation

### Zero-Shot Classification

```bash
python evaluation/zero_shot/zero_shot_CAMELYON16.py \
    --image_emb /path/to/embeddings.npy \
    --slide_ids /path/to/slide_ids.npy \
    --text_emb  /path/to/text_embeddings/CAMELYON16.npy \
    --csv_path  /path/to/CAM16.csv \
    --output_dir ./results/zero_shot
```

Scripts available for: `CAMELYON16`, `CAMELYON17`, `PANDA`, `TCGA-GBMLGG`, `UNITOPATHO`

### Cross-Modal Retrieval

```bash
# Image-to-image
python evaluation/cross_modality/image_to_image.py \
    --dataset CAMELYON16 \
    --image_emb /path/to/embeddings.npy \
    --slide_ids /path/to/slide_ids.npy \
    --csv_path  /path/to/CAM16.csv \
    --out_json  ./results/image_to_image.json

# Image-to-text
python evaluation/cross_modality/image_to_text.py \
    --dataset CAMELYON16 \
    --image_emb /path/to/embeddings.npy \
    --slide_ids /path/to/slide_ids.npy \
    --text_root /path/to/text_embeddings \
    --csv_path  /path/to/CAM16.csv \
    --out_json  ./results/image_to_text.json

# Text-to-image
python evaluation/cross_modality/text_to_image.py \
    --dataset CAMELYON16 \
    --image_emb /path/to/embeddings.npy \
    --slide_ids /path/to/slide_ids.npy \
    --text_root /path/to/text_embeddings \
    --csv_path  /path/to/CAM16.csv \
    --save_csv  ./results/text_to_image.csv
```

### Open Vocabulary Retrieval

```bash
python evaluation/open_vocab/evaluate_openvocab.py \
    --image_emb  /path/to/embeddings.npy \
    --slide_ids  /path/to/slide_ids.npy \
    --emb_root   /path/to/text_embeddings \
    --csv_path   /path/to/CAM16.csv \
    --dataset    CAMELYON16 \
    --model      plip \
    --output_dir ./results/open_vocab
```

### Paraphrase Robustness

```bash
python evaluation/paraphrase_robustness/evaluate_paraphrase.py \
    --image_emb  /path/to/embeddings.npy \
    --slide_ids  /path/to/slide_ids.npy \
    --pool_emb   /path/to/paraphrase_pool.npy \
    --csv_path   /path/to/CAM16.csv \
    --dataset    CAMELYON16 \
    --model      plip \
    --output_dir ./results/paraphrase_robustness
```

### Alignment Score

```bash
python evaluation/zero_shot/alignment.py \
    --image_emb  /path/to/embeddings.npy \
    --slide_ids  /path/to/slide_ids.npy \
    --text_emb   /path/to/text_embeddings/CAMELYON16.npy \
    --csv_path   /path/to/CAM16.csv \
    --dataset    CAMELYON16 \
    --model      plip \
    --output_dir ./results/alignment
```

