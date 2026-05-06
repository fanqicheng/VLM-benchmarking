from __future__ import annotations

import os
import sys
import zipfile
import subprocess
import importlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
from torchvision import transforms

import open_clip
from transformers import (
    AutoModel as HFAutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    XLMRobertaTokenizer,
)

# transformers imports
try:
    from transformers import AutoProcessor
    HAS_AUTO_PROCESSOR = True
except ImportError:
    HAS_AUTO_PROCESSOR = False
    print("Warning: AutoProcessor not available, using AutoImageProcessor")

try:
    from transformers import AutoImageProcessor
    HAS_AUTO_IMAGE_PROCESSOR = True
except ImportError:
    HAS_AUTO_IMAGE_PROCESSOR = False

# Third-party pathology model imports
try:
    import timm_ctp
except ImportError:
    pass

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
from huggingface_hub import hf_hub_download
from open_clip.pretrained import download_pretrained_from_hf

import musk as musk_pkg
from musk import modeling, utils  # noqa — triggers timm model registration

from text_adapter_UPDATE import attach_text_adapter


@dataclass
class FMWrapper:
    """FM wrapper"""
    name: str
    model: torch.nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    has_text_encoder: bool
    encode_image: Callable[[torch.Tensor], torch.Tensor]
    encode_text: Optional[Callable[[List[str]], torch.Tensor]] = None
    image_size: Optional[int] = None
    precision: torch.dtype = torch.float32

FM_REGISTRY: Dict[str, Callable[..., FMWrapper]] = {}


def load_fm(name: str, device: str = "cuda:0", **kwargs) -> FMWrapper:

    if name not in FM_REGISTRY:
        raise KeyError(f"Unknown FM name: {name}. Available: {list(FM_REGISTRY.keys())}")
    return FM_REGISTRY[name](device=device, **kwargs)


def register_fm(name: str):
    def decorator(fn):
        if name in FM_REGISTRY:
            raise ValueError(f"FM name '{name}' already registered")
        FM_REGISTRY[name] = fn
        return fn
    return decorator


def _load_processor(repo_id: str):
    if HAS_AUTO_PROCESSOR:
        try:
            from transformers import AutoProcessor
            return AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        except:
            pass
    
    if HAS_AUTO_IMAGE_PROCESSOR:
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(repo_id, trust_remote_code=True)
    
    raise ImportError("Cannot import AutoProcessor or AutoImageProcessor from transformers")

 
def _pip_install(*packages: str):
    import subprocess, sys
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + list(packages)
    print(f"[install] pip install {' '.join(packages)}")
    subprocess.check_call(cmd)
 
 
def _ensure_packages(**pkg_map):
    import importlib
    for import_name, pip_name in pkg_map.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"[install] '{import_name}' not found, installing '{pip_name}' ...")
            _pip_install(pip_name)
 
 
def _ensure_musk():
    import importlib
    try:
        importlib.import_module("musk")
        return  
    except ImportError:
        pass

    import subprocess, sys, os, urllib.request, zipfile
    zip_path = "/tmp/musk_install.zip"
    musk_dir = "/tmp/MUSK-main"

    if not os.path.exists(musk_dir):
        print("[install] Downloading MUSK source from GitHub ...")
        urllib.request.urlretrieve(
            "https://github.com/lilab-stanford/MUSK/archive/refs/heads/main.zip",
            zip_path
        )
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("/tmp")
        print("[install] Extracted to /tmp/MUSK-main")

    req_file = os.path.join(musk_dir, "requirements.txt")
    if os.path.exists(req_file):
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", "-r", req_file])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", "-e", musk_dir])
    print("[install] MUSK installed")


# PLIP (vinid/plip)
@register_fm("plip")
def load_plip(device: str = "cuda:0") -> FMWrapper:
    repo_id = "vinid/plip"

    processor = CLIPProcessor.from_pretrained(repo_id)
    model = CLIPModel.from_pretrained(repo_id)

    model = model.to(device)
    model.eval()

    def _to_tensor(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs

        if hasattr(outputs, "text_embeds"):
            return outputs.text_embeds

        if hasattr(outputs, "image_embeds"):
            return outputs.image_embeds

        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output

        raise RuntimeError(f"Cannot extract tensor from: {type(outputs)}")

    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        inputs = processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = model.get_image_features(pixel_values=img_batch.to(device))
            feat = _to_tensor(outputs)

        feat = F.normalize(feat, dim=-1)
        return feat

    def _encode_text(texts: List[str]) -> torch.Tensor:
        all_embeddings = []

        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)

            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
                feat = _to_tensor(outputs)

            feat = F.normalize(feat, dim=-1)
            all_embeddings.append(feat)

        return torch.cat(all_embeddings, dim=0)

    fm = FMWrapper(
        name="plip",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=True,
        encode_image=_encode_image,
        encode_text=_encode_text,
        image_size=224,
    )

    attach_text_adapter(
    fm,
    tokenizer=processor,
    device=device,
    )

    return fm



# KEEP (Astaxanthin/KEEP)
@register_fm("keep")
def load_keep(device: str = "cuda:0") -> FMWrapper:
    
    
    repo_id = "Astaxanthin/KEEP"
    model = AutoModel.from_pretrained(
        repo_id, 
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None  
    )

    if device != "cpu":
        model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return transform(pil_img.convert('RGB'))

    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = model.encode_image(img_batch)
        return feat

    def _encode_text(texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            token_input = tokenizer(
                texts,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            token_input = {k: v.to(device) for k, v in token_input.items()}
            
            feat = model.encode_text(token_input)
        return feat

    fm = FMWrapper(
        name="keep",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )

    attach_text_adapter(
    fm,
    tokenizer=tokenizer,
    device=device,
    )

    return fm



# BioMedCLIP v2 
@register_fm("biomedclip-v2")
def load_biomedclip_v2(device: str = "cuda:0") -> FMWrapper:

    # repo_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    oc_repo_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    hf_repo_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    model, _, preprocess_fn = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained=None,
    )

    ckpt = download_pretrained_from_hf(hf_repo_id)

    state_dict = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    tokenizer = open_clip.get_tokenizer(oc_repo_id)

    model = model.to(device).eval()

    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return preprocess_fn(pil_img.convert("RGB"))
 
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = model.encode_image(img_batch)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    fm = FMWrapper(
        name="biomedclip-v2",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )

    attach_text_adapter(
        fm,
        tokenizer=tokenizer,
        device=device,
    )

    return fm


# PathGen-CLIP / PathGen-CLIP-L (jamessyx/)
def _download_pathgen_weights(repo_id: str, filename: str, save_path: str):
    """auto download PathGen-CLIP weight"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[PathGen-CLIP] Downloading {filename} from {repo_id} ...")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(save_path))
    print(f"[PathGen-CLIP] ✅ Done: {save_path}")
 
 
@register_fm("pathgen-clip")
def load_pathgen_clip(
    device: str = "cuda:0",
    ckpt_path: Optional[str] = None,
) -> FMWrapper:

 
    if ckpt_path is None:
        import os
        ckpt_path = os.path.expanduser("~/.cache/pathgen_clip/pathgenclip.pt")
        if not os.path.exists(ckpt_path):
            _download_pathgen_weights("jamessyx/PathGen-CLIP", "pathgenclip.pt", ckpt_path)

    model, _, preprocess_fn = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=ckpt_path
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
 
    model = model.to(device)
    model.eval()
 
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return preprocess_fn(pil_img.convert("RGB"))
 
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = model.encode_image(img_batch)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    fm = FMWrapper(
        name="pathgen-clip",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )

    attach_text_adapter(
    fm,
    tokenizer=tokenizer,
    device=device,
    )

    return fm


# MUSK (xiangjx/musk) — Nature 2025
@register_fm("musk")
def load_musk(device: str = "cuda:0") -> FMWrapper:
    _ensure_musk()

    model = create_model("musk_large_patch16_384")

    utils.load_model_and_may_interpolate(
        "hf_hub:xiangjx/musk",
        model,
        "model|module",
        ""
    )
    model.to(device=device, dtype=torch.float16)
    model.eval()

    _transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384, interpolation=3, antialias=True),
        torchvision.transforms.CenterCrop((384, 384)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_INCEPTION_MEAN,
            std=IMAGENET_INCEPTION_STD
        ),
    ])
 
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return _transform(pil_img.convert("RGB"))

    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            vision_cls, _ = model(
                image=img_batch.to(device, dtype=torch.float16),
                with_head=False,
                out_norm=True,
                ms_aug=True,
                return_global=True,
            )
        return vision_cls.float()
 
    def _encode_text(texts: List[str]) -> torch.Tensor:
        

        pkg_dir = os.path.dirname(musk_pkg.__file__)
        spm_path = os.path.join(pkg_dir, "models", "tokenizer.spm")
        tokenizer = XLMRobertaTokenizer(spm_path)

        encoded = tokenizer(
            texts,
            max_length=100,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        txt_ids = encoded["input_ids"].to(device)
        pad_mask = (encoded["attention_mask"] == 0).long().to(device)

        model.float()
        with torch.inference_mode():
            _, text_cls = model(
                text_description=txt_ids,
                padding_mask=pad_mask,
                with_head=False,
                out_norm=True,
                ms_aug=False,
                return_global=True,
            )
        model.half()
        return text_cls.float()

    return FMWrapper(
        name="musk",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=True,
        encode_image=_encode_image,
        encode_text=_encode_text,
        image_size=384,
        precision=torch.float16,
    )
 
 

# CONCH (MahmoodLab - CoCa-based pathology VLM)

@register_fm("conch")
def load_conch(device: str = "cuda:0") -> FMWrapper:
    

    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
    tokenizer = get_tokenizer()
    model = model.to(device)
    model.eval()

    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return preprocess(pil_img)

    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = model.encode_image(img_batch)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    def _encode_text(texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = tokenizer(texts, padding=True, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)
            feat = model.encode_text(input_ids)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    return FMWrapper(
        name="conch",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=True,
        encode_image=_encode_image,
        encode_text=_encode_text,
        image_size=224,
    )


# Patho-CLIP-L 

@register_fm("patho-clip")
def load_patho_clip(
    device: str = "cuda:0",
    ckpt_path: Optional[str] = None,
) -> FMWrapper:
    
    _ensure_packages(open_clip="open_clip_torch")
 
    if ckpt_path is not None:
        model, _, preprocess_fn = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=ckpt_path
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    else:
        ckpt_path = hf_hub_download(
            repo_id="WenchuanZhang/Patho-CLIP-L",
            filename="Patho-CLIP-L.pt",
            local_dir=os.path.expanduser("~/.cache/patho_clip"),
        )
        model, _, preprocess_fn = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=ckpt_path
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
 
    model = model.to(device).eval()
 
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return preprocess_fn(pil_img.convert("RGB"))
 
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = model.encode_image(img_batch)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    fm = FMWrapper(
        name="patho-clip",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )

    attach_text_adapter(
    fm,
    tokenizer=tokenizer,
    device=device,
    )

    return fm



# MI-Zero (mahmoodlab/MI-Zero) 

def _ensure_timm_ctp():

    print("[MI-Zero] Installing timm_ctp ...")

    zip_path = "/tmp/mizero_install.zip"
    mizero_dir = "/tmp/MI-Zero-main"
 
    if not os.path.exists(mizero_dir):
        print("[MI-Zero] Downloading MI-Zero source ...")
        urllib.request.urlretrieve(
            "https://github.com/mahmoodlab/MI-Zero/archive/refs/heads/main.zip",
            zip_path,
        )
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("/tmp")
 
    timm_ctp_tar = os.path.join(mizero_dir, "assets", "timm_ctp.tar")
    if os.path.exists(timm_ctp_tar):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--quiet", "--no-deps", timm_ctp_tar,
        ])
        print("[MI-Zero] ✅ timm_ctp installed")
    else:
        print("[MI-Zero] ⚠️  timm_ctp.tar not found, skipping (may cause errors)")
 
 
@register_fm("mi-zero")
def load_mi_zero(
    device: str = "cuda:0",
    ckpt_path: Optional[str] = None,
    text_encoder: str = "bioclinicalbert",  
) -> FMWrapper:


    if ckpt_path is None:
        raise ValueError(

        )

    _ensure_timm_ctp()

    print(f"[MI-Zero] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "", 1)
        new_state_dict[new_key] = v
    state_dict = new_state_dict

    try:
        
        visual_model = timm_ctp.create_model(
            "swin_tiny_patch4_window7_224",
            embed_layer=timm_ctp.models.swin_transformer.ConvStem,
            pretrained=False,
            num_classes=0,
            img_size=448,  
        )
    except Exception as e:
        print(f"[MI-Zero] timm_ctp failed ({e}), falling back to timm Swin")
        import timm
        visual_model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0,
            img_size=448,
        )

    visual_sd = {
        k.replace("visual.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("visual.")
    }
 
    if visual_sd:
        missing, unexpected = visual_model.load_state_dict(visual_sd, strict=False)
        print(f"[MI-Zero] Visual encoder loaded "
              f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
    else:
        print("[MI-Zero] ⚠️  No 'visual.*' keys found in checkpoint")
 
    visual_model = visual_model.to(device).eval()

    if text_encoder == "bioclinicalbert":
        text_model_id = "emilyalsentzer/Bio_ClinicalBERT"
    elif text_encoder == "pubmedbert":
        text_model_id = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    else:
        raise ValueError(f"text_encoder must be 'bioclinicalbert' or 'pubmedbert', got: {text_encoder}")
 
    print(f"[MI-Zero] Loading text encoder: {text_model_id}")
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_model = HFAutoModel.from_pretrained(text_model_id).to(device).eval()

    _transform = T.Compose([
        T.Resize(448, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])
 
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return _transform(pil_img.convert("RGB"))
 
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
  
        with torch.no_grad():
            feat = visual_model.forward_features(img_batch)
            if feat.dim() == 4:
                feat = feat.mean(dim=[1, 2]) 
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)       
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat
 
    def _encode_text(texts: List[str]) -> torch.Tensor:
      
        enc = text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = text_model(**enc)
            feat = out.last_hidden_state[:, 0]  
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat
 
    return FMWrapper(
        name="mi-zero",
        model=visual_model,
        preprocess=_preprocess,
        has_text_encoder=True,
        encode_image=_encode_image,
        encode_text=_encode_text,
        image_size=448,
        precision=torch.float32,
    )


# QuiltNet-B-16
@register_fm("quiltnet")
def load_quiltnet(device: str = "cuda:0") -> FMWrapper:
    _ensure_packages(open_clip="open_clip_torch")

    model, _, preprocess_fn = open_clip.create_model_and_transforms(
        'hf-hub:wisdomik/QuiltNet-B-16'
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-16')
    model = model.to(device).eval()

    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return preprocess_fn(pil_img.convert("RGB"))

    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = model.encode_image(img_batch)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    def _encode_text(texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = tokenizer(texts).to(device)
            feat = model.encode_text(tokens)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    fm = FMWrapper(
        name="quiltnet",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=True,
        encode_image=_encode_image,
        encode_text=_encode_text,
        image_size=224,
    )
    return fm

if __name__ == "__main__":
    print("Available models:", list(FM_REGISTRY.keys()))