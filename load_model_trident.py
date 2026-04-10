#!/usr/bin/env python3
"""
load_model.py - 统一的模型加载接口（兼容旧版本 transformers）
"""

from dataclasses import dataclass
from typing import Callable, Optional, Dict, List
# for text adapter ADD
from text_adapter_trident import attach_text_adapter

from PIL import Image
import torch

# 兼容旧版本 transformers
try:
    from transformers import AutoProcessor, AutoModel
    HAS_AUTO_PROCESSOR = True
except ImportError:
    from transformers import AutoModel
    HAS_AUTO_PROCESSOR = False
    print("⚠️  Warning: AutoProcessor not available, using AutoImageProcessor")

try:
    from transformers import AutoImageProcessor
    HAS_AUTO_IMAGE_PROCESSOR = True
except ImportError:
    HAS_AUTO_IMAGE_PROCESSOR = False


@dataclass
class FMWrapper:
    """统一后的 FM 接口封装"""
    name: str
    model: torch.nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    has_text_encoder: bool
    encode_image: Callable[[torch.Tensor], torch.Tensor]
    encode_text: Optional[Callable[[List[str]], torch.Tensor]] = None
    image_size: Optional[int] = None
    precision: torch.dtype = torch.float32


# 注册表：模型名 → 加载函数
FM_REGISTRY: Dict[str, Callable[..., FMWrapper]] = {}


def register_fm(name: str):
    """装饰器：把 loader 函数注册到 FM_REGISTRY 里"""
    def decorator(fn):
        if name in FM_REGISTRY:
            raise ValueError(f"FM name '{name}' already registered")
        FM_REGISTRY[name] = fn
        return fn
    return decorator


def _load_processor(repo_id: str):
    """加载处理器（兼容不同版本）"""
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


# ============================================================
# 统一依赖安装工具
# ============================================================
 
def _pip_install(*packages: str):
    """在运行时自动 pip install 缺失的包"""
    import subprocess, sys
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + list(packages)
    print(f"[install] pip install {' '.join(packages)}")
    subprocess.check_call(cmd)
 
 
def _ensure_packages(**pkg_map):
    """
    检查并安装缺失的包
    pkg_map: {import_name: pip_package_name}
    例如: _ensure_packages(open_clip="open_clip_torch", timm="timm")
    """
    import importlib
    for import_name, pip_name in pkg_map.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"[install] '{import_name}' not found, installing '{pip_name}' ...")
            _pip_install(pip_name)
 
 
def _ensure_musk():
    """安装 MUSK 官方包（用 urllib 下载 zip，无需 git）"""
    import importlib
    try:
        importlib.import_module("musk")
        return  # 已安装，跳过
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
    print("[install] ✅ MUSK installed")


 
# ============================================================
# mSTAR+ (JHU-CAML/mstar_plus)
# ============================================================
@register_fm("mstar")
def load_mstar(device: str = "cuda:0") -> FMWrapper:
    """
    加载 mSTAR 模型（按照官方instruction）
    Official repo: Wangyh/mSTAR
    Paper: A Multimodal Knowledge-enhanced Whole-slide Pathology Foundation Model
    """
    import timm
    from torchvision import transforms
    
    # 使用官方推荐的方式加载
    model = timm.create_model(
        'hf-hub:Wangyh/mSTAR',
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True
    )
    
    # 移动到指定设备
    model = model.to(device)
    model.eval()
    
    # 官方指定的图像预处理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        """预处理单张图像"""
        return transform(pil_img.convert('RGB'))
    
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        """编码图像批次"""
        with torch.no_grad():
            # mSTAR 是 ViT 架构，使用 forward_features 获取特征
            feat = model.forward_features(img_batch)
            
            # ViT 通常返回 [B, N, D] 的特征，取 CLS token (第一个 token)
            if feat.dim() == 3:
                feat = feat[:, 0]  # 取 CLS token
            
            # 归一化
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        
        return feat
    
    return FMWrapper(
        name="mstar",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,  # mSTAR 是 vision-only 模型
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )
# ============================================================
# Hibou-B (HistAI - DINOv2-based Vision Transformer)
# ============================================================

@register_fm("hibou-b")
def load_hibou_b(device: str = "cuda:0") -> FMWrapper:
    """
    加载 Hibou-B 模型(按照官方instruction)
    Official repo: https://github.com/HistAI/hibou
    Paper: A Foundational Vision Transformer for digital pathology
    Architecture: DINOv2-based ViT with registers
    """
    from torchvision import transforms
    
    repo_id = "histai/hibou-b"
    
    # 加载 processor 和 model (需要 trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    # 使用官方推荐的图像预处理
    # 根据 DINOv2 的标准,通常使用 224x224 或 518x518
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        """预处理单张图像"""
        return transform(pil_img.convert('RGB'))
    
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        """
        编码图像批次
        DINOv2 架构通常返回 [B, N, D] 的特征,其中:
        - B: batch size
        - N: number of tokens (包括 CLS token 和可能的 register tokens)
        - D: embedding dimension
        """
        with torch.no_grad():
            # 构建输入字典
            inputs = {"pixel_values": img_batch}
            outputs = model(**inputs)
            
            # 根据输出类型提取特征
            if hasattr(outputs, 'last_hidden_state'):
                # DINOv2 通常使用 CLS token (第一个 token)
                feat = outputs.last_hidden_state[:, 0]
            elif hasattr(outputs, 'pooler_output'):
                feat = outputs.pooler_output
            else:
                # 如果是直接的 tensor 输出
                if isinstance(outputs, torch.Tensor):
                    if outputs.dim() == 3:
                        feat = outputs[:, 0]  # 取 CLS token
                    else:
                        feat = outputs
                else:
                    # 尝试从 tuple/list 中获取
                    feat = outputs[0]
                    if feat.dim() == 3:
                        feat = feat[:, 0]
            
            # 归一化
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        
        return feat
    
    return FMWrapper(
        name="hibou-b",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,  # Hibou-B 是 vision-only 模型
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )

# ============================================================
# Phikon-v2 (Owkin - 病理学 ViT 模型)
# ============================================================

@register_fm("phikon-v2")
def load_phikon_v2(device: str = "cuda:0") -> FMWrapper:
    """
    加载 Phikon-v2 模型（Owkin）
    Official repo: owkin/phikon-v2
    Paper: HistoSSLscaling - A foundation model for histopathology
    """
    repo_id = "owkin/phikon-v2"
    
    # 加载 processor 和 model
    processor = _load_processor(repo_id)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        """预处理单张图像"""
        inputs = processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        """编码图像批次 - 提取 CLS token"""
        with torch.no_grad():
            inputs = {"pixel_values": img_batch}
            outputs = model(**inputs)
            
            # 根据官方文档，使用 last_hidden_state 的第一个 token (CLS token)
            feat = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, 1024]
            
            # 归一化
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    return FMWrapper(
        name="phikon-v2",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,  # Phikon-v2 是 vision-only 模型
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,  # 标准 ViT 输入尺寸
    )

# ============================================================
# PLIP (vinid/plip)
# ============================================================
@register_fm("plip")
def load_plip(device: str = "cuda:0") -> FMWrapper:
    """
    Stable PLIP loader with safe tensor extraction
    Works for vinid/plip and avoids all HF inconsistencies
    """

    from transformers import CLIPProcessor, CLIPModel
    import torch
    import torch.nn.functional as F
    from typing import List
    from PIL import Image

    repo_id = "vinid/plip"

    processor = CLIPProcessor.from_pretrained(repo_id)
    model = CLIPModel.from_pretrained(repo_id)

    model = model.to(device)
    model.eval()

    # ============================================================
    # 🔥 SAFE EXTRACTOR (核心：统一所有模型输出)
    # ============================================================
    def _to_tensor(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs

        if hasattr(outputs, "text_embeds"):
            return outputs.text_embeds

        if hasattr(outputs, "image_embeds"):
            return outputs.image_embeds

        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output

        raise RuntimeError(f"❌ Cannot extract tensor from: {type(outputs)}")

    # ============================================================
    # Image Preprocess
    # ============================================================
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        inputs = processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    # ============================================================
    # Image Encoder
    # ============================================================
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = model.get_image_features(pixel_values=img_batch.to(device))
            feat = _to_tensor(outputs)

        feat = F.normalize(feat, dim=-1)
        return feat

    # ============================================================
    # Text Encoder
    # ============================================================
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
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )

    attach_text_adapter(
    fm,
    tokenizer=processor,
    device=device,
    )

    return fm


# ============================================================
# KEEP (Astaxanthin/KEEP)
# ============================================================

@register_fm("keep")
def load_keep(device: str = "cuda:0") -> FMWrapper:
    """加载 KEEP 模型（根据官方 instruction）"""
    from torchvision import transforms
    from transformers import AutoTokenizer
    
    repo_id = "Astaxanthin/KEEP"
    
    # 加载模型和 tokenizer
    # 强制加载到 CPU，避免 device 参数冲突
    model = AutoModel.from_pretrained(
        repo_id, 
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None  # 不自动分配设备
    )
    
    # 手动移到目标设备
    if device != "cpu":
        model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    
    # 官方的图像预处理
    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        """预处理单张图像"""
        return transform(pil_img.convert('RGB'))

    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        """编码图像批次"""
        with torch.no_grad():
            # KEEP 有专门的 encode_image 方法
            feat = model.encode_image(img_batch)
            # 已经归一化，不需要再次归一化
        return feat

    def _encode_text(texts: List[str]) -> torch.Tensor:
        """编码文本"""
        with torch.no_grad():
            # 使用 tokenizer
            token_input = tokenizer(
                texts,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            token_input = {k: v.to(device) for k, v in token_input.items()}
            
            # KEEP 有专门的 encode_text 方法
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

# ============================================================
# CTransPath (Transformer-based Contrastive Learning)
# ============================================================
@register_fm("ctranspath")
def load_ctranspath(device: str = "cuda:0") -> FMWrapper:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    
    print("🔵 Loading CTransPath (Custom Swin with modified downsample) ...")

    # -------------------------------------------------------
    # CTransPath 使用了自定义的 Swin Transformer
    # 主要区别:
    # 1. Patch embed 使用卷积序列
    # 2. Downsample 的输入维度是 4 倍
    # -------------------------------------------------------
    
    ckpt_path = "/Data3/shangke/model/CTransPath/ctranspath.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    
    # 去除 "model." 前缀
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("model."):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    
    # 删除分类头
    to_remove = [k for k in state_dict.keys() if k.startswith("head")]
    for k in to_remove:
        del state_dict[k]
    
    print("📊 Checkpoint info:")
    print(f"   - Total keys: {len(state_dict)}")
    
    # -------------------------------------------------------
    # 方案: 使用 timm 的 Swin,但修改 downsample 层来匹配
    # -------------------------------------------------------
    from timm.models.swin_transformer import SwinTransformer
    
    # 先创建标准模型
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        num_classes=0
    )
    
    # 加载权重,允许部分匹配
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"\n⚠️  Initial load (with mismatches):")
    print(f"   - Missing: {len(missing)} keys")
    print(f"   - Unexpected: {len(unexpected)} keys")
    
    # -------------------------------------------------------
    # 手动修复 downsample 层
    # -------------------------------------------------------
    print("\n🔧 Fixing downsample layers...")
    
    for stage_idx in range(3):  # 只有前3个stage有downsample
        stage = model.layers[stage_idx]
        
        if hasattr(stage, 'downsample') and stage.downsample is not None:
            # 获取原始维度
            old_downsample = stage.downsample
            
            # 从 checkpoint 获取正确的维度
            norm_key = f"layers.{stage_idx}.downsample.norm.weight"
            reduction_key = f"layers.{stage_idx}.downsample.reduction.weight"
            
            if norm_key in state_dict and reduction_key in state_dict:
                # 获取正确的维度
                norm_dim = state_dict[norm_key].shape[0]
                reduction_weight = state_dict[reduction_key]
                out_dim, in_dim = reduction_weight.shape
                
                print(f"   Stage {stage_idx}: {in_dim} -> {out_dim} (norm: {norm_dim})")
                
                # 创建新的 downsample 层
                class CustomDownsample(nn.Module):
                    def __init__(self, in_dim, out_dim):
                        super().__init__()
                        self.norm = nn.LayerNorm(in_dim)
                        self.reduction = nn.Linear(in_dim, out_dim, bias=False)
                    
                    def forward(self, x):
                        # x: [B, H, W, C]
                        x = self.norm(x)
                        # Swin的downsample会合并patches: [B, H, W, C] -> [B, H/2, W/2, 4*C]
                        B, H, W, C = x.shape
                        x = x.view(B, H // 2, 2, W // 2, 2, C)
                        x = x.permute(0, 1, 3, 4, 2, 5).contiguous()
                        x = x.view(B, H // 2, W // 2, 4 * C)
                        x = self.reduction(x)
                        return x
                
                # 替换 downsample
                new_downsample = CustomDownsample(in_dim, out_dim)
                stage.downsample = new_downsample
                
                # 加载对应的权重
                stage.downsample.norm.weight.data.copy_(state_dict[norm_key])
                stage.downsample.norm.bias.data.copy_(state_dict[f"layers.{stage_idx}.downsample.norm.bias"])
                stage.downsample.reduction.weight.data.copy_(state_dict[reduction_key])
    
    print("\n✅ Downsample layers fixed!")
    
    model = model.to(device).eval()

    # 预处理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    def _preprocess(pil_img: Image.Image):
        return transform(pil_img.convert("RGB"))

    def _encode_image(img_batch: torch.Tensor):
        with torch.no_grad():
            feat = model.forward_features(img_batch)
            # Swin Transformer 输出: [B, H, W, C]
            if feat.dim() == 4:
                feat = feat.mean(dim=[1, 2])  # 平均池化 -> [B, C]
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)       # 如果是 [B, N, C] -> [B, C]
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    return FMWrapper(
        name="ctranspath",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )
# ============================================================
# Prov-GigaPath (Microsoft - 公开可用)
# ============================================================

@register_fm("prov-gigapath")
def load_prov_gigapath(device: str = "cuda:0") -> FMWrapper:
    """
    加载 Prov-GigaPath 模型（按照官方instruction）
    Official repo: prov-gigapath/prov-gigapath
    Paper: A whole-slide foundation model for digital pathology from real-world data
    Architecture: ViT-giant with 1.1B parameters
    """
    import timm
    from torchvision import transforms
    
    # 按照官方文档创建模型
    tile_encoder = timm.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        pretrained=True
    )
    
    # 移动到指定设备
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    # 使用官方推荐的 transform
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        """预处理单张图像"""
        return transform(pil_img.convert('RGB'))
    
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        """
        编码图像批次
        输出: tile-level embeddings (1536维)
        """
        with torch.no_grad():
            output = tile_encoder(img_batch)  # 直接调用模型
            
            # 如果输出是多维的，取第一个维度（通常是 CLS token）
            if output.dim() == 3:
                output = output[:, 0]  # 取 CLS token
            
            # 归一化
            embedding = torch.nn.functional.normalize(output, p=2, dim=-1)
        
        return embedding
    
    return FMWrapper(
        name="prov-gigapath",
        model=tile_encoder,
        preprocess=_preprocess,
        has_text_encoder=False,  # Prov-GigaPath 是 vision-only 模型
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )



# ============================================================
# BioMedCLIP v2 — 官方 open_clip 方式（推荐替代原 biomedclip）
# ============================================================

@register_fm("biomedclip-v2")
def load_biomedclip_v2(device: str = "cuda:0") -> FMWrapper:
    """
    加载 BioMedCLIP（官方 open_clip 方式，Microsoft）
    HF repo : microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    Paper   : BiomedCLIP (arXiv:2303.00915)
    Arch    : ViT-B/16 (image) + PubMedBERT (text)，open_clip 框架
    License : MIT（完全开放，无需登录）
 
    与原 biomedclip 的区别：
      - 原版 (biomedclip)  使用 transformers AutoModel → 不能直接调用
        encode_image / encode_text，需手动取 pooler_output
      - 此版 (biomedclip-v2) 使用官方推荐的 open_clip API → 接口更简洁准确
 
    Install : pip install open_clip_torch>=2.23.0
    """
    import open_clip
 
    repo_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # model, preprocess_fn = open_clip.create_model_from_pretrained(repo_id)
    # tokenizer = open_clip.get_tokenizer(repo_id)
    
    oc_repo_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    hf_repo_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    model, _, preprocess_fn = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained=None,
    )

    from open_clip.pretrained import download_pretrained_from_hf
    ckpt = download_pretrained_from_hf(hf_repo_id)

    state_dict = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    tokenizer = open_clip.get_tokenizer(oc_repo_id)

    model = model.to(device).eval()
    # model = model.to(device)
    # model.eval()
 
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return preprocess_fn(pil_img.convert("RGB"))
 
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = model.encode_image(img_batch)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat
 
    # def _encode_text(texts: List[str]) -> torch.Tensor:
    #     tokens = tokenizer(texts).to(device)
    #     with torch.no_grad():
    #         feat = model.encode_text(tokens)
    #         feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
    #     return feat
 
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

# ============================================================
# PathGen-CLIP / PathGen-CLIP-L (jamessyx/)
# ============================================================
#
# HF 可用性：
#   PathGen-CLIP (ViT-B/16)
#     页面  : https://huggingface.co/jamessyx/PathGen-CLIP
#     状态  : ⚠️  Gated（需登录同意条款）
#     但权重有公开直链，无需登录可直接下载：
#     https://pub-7a38cc906afa44a4a01533c288d0b1af.r2.dev/pathgenclip.pt
#
#   PathGen-CLIP-L (ViT-L/14)
#     页面  : https://huggingface.co/jamessyx/PathGen-CLIP-L
#     状态  : ⚠️  Gated（需登录同意条款）
#     权重直链（公开）：
#     https://pub-7a38cc906afa44a4a01533c288d0b1af.r2.dev/pathgenclip_l.pt
#
# 官方加载方式（按 README）：
#   open_clip.create_model_and_transforms('ViT-B-16', pretrained='path/to/pathgenclip.pt')
#   open_clip.create_model_and_transforms('ViT-L-14', pretrained='path/to/pathgenclip_l.pt')
# ============================================================
 
def _download_pathgen_weights(repo_id: str, filename: str, save_path: str):
    """自动下载 PathGen-CLIP 权重"""
    import os, urllib.request
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[PathGen-CLIP] Downloading {filename} from {repo_id} ...")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(save_path))
    print(f"[PathGen-CLIP] ✅ Done: {save_path}")
 
 
@register_fm("pathgen-clip")
def load_pathgen_clip(
    device: str = "cuda:0",
    ckpt_path: Optional[str] = None,
) -> FMWrapper:
    """
    加载 PathGen-CLIP (ViT-B/16)
    Paper   : PathGen-1.6M (arXiv:2407.00203)
    Arch    : ViT-B/16，open_clip 框架
    License : CC-BY-2.0
 
    权重会自动下载到 ~/.cache/pathgen_clip/pathgenclip.pt
    也可手动指定 ckpt_path 跳过下载。
    """
    import open_clip
 
    if ckpt_path is None:
        import os
        ckpt_path = os.path.expanduser("~/.cache/pathgen_clip/pathgenclip.pt")
        if not os.path.exists(ckpt_path):
            _download_pathgen_weights("jamessyx/PathGen-CLIP", "pathgenclip.pt", ckpt_path)
 
    # 官方加载方式
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
 
    # def _encode_text(texts: List[str]) -> torch.Tensor:
    #     tokens = tokenizer(texts).to(device)
    #     with torch.no_grad():
    #         feat = model.encode_text(tokens)
    #         feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
    #     return feat
 
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
 
# ============================================================
# Virchow2 (paige-ai/Virchow2)
# ============================================================

@register_fm("virchow2")
def load_virchow2(device: str = "cuda:0") -> FMWrapper:
    """
    加载 Virchow2 模型（按照官方instruction）
    Official repo: paige-ai/Virchow2
    Paper: Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology
    Architecture: ViT-H/14 with 632M parameters
    """
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked
    
    # 按照官方文档创建模型，需要指定 MLP layer 和 activation function
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU
    )
    
    # 移动到指定设备
    model = model.to(device)
    model.eval()
    
    # 使用官方推荐的 transform
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        """预处理单张图像"""
        return transforms(pil_img.convert('RGB'))
    
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        """
        编码图像批次
        输出: 1 x 261 x 1280 (class token + 4 register tokens + 256 patch tokens)
        最终embedding: 1 x 2560 (class token + mean of patch tokens)
        """
        # 使用混合精度推理（官方推荐）
        use_fp16 = device.startswith("cuda")
        
        if use_fp16:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(img_batch)  # size: B x 261 x 1280
        else:
            with torch.no_grad():
                output = model(img_batch)
        
        # 提取 class token 和 patch tokens
        class_token = output[:, 0]      # size: B x 1280
        patch_tokens = output[:, 5:]    # size: B x 256 x 1280 (跳过4个register tokens)
        
        # 拼接 class token 和 patch tokens 的平均值
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: B x 2560
        
        # 转换为 fp16 以提高效率（可选）
        if use_fp16:
            embedding = embedding.to(torch.float16)
        
        # 归一化
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    return FMWrapper(
        name="virchow2",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,  # Virchow2 是 vision-only 模型
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
        precision=torch.float16,
    )

# ============================================================
# MUSK (xiangjx/musk) — Nature 2025
# 完全按照官方 README instruction 实现
# ============================================================
 
@register_fm("musk")
def load_musk(device: str = "cuda:0") -> FMWrapper:
    """
    加载 MUSK 模型（完全按照官方 README）
    HF repo : xiangjx/musk  (Gated，需登录同意条款)
    Paper   : Nature 2025 — A Vision-Language Foundation Model for Precision Oncology
    GitHub  : https://github.com/lilab-stanford/MUSK
    Arch    : musk_large_patch16_384，image=384×384，dim=2048
    License : CC-BY-NC-ND 4.0
 
    依赖安装（自动）：
        pip install -r MUSK/requirements.txt
        pip install -e MUSK/
 
    前置条件（需手动）：
        huggingface-cli login   # 需通过 xiangjx/musk gated 申请
 
    官方加载方式：
        from musk import utils, modeling
        from timm.models import create_model
        model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
        model.to(device="cuda", dtype=torch.float16)
        model.eval()
    """
    # 自动安装 musk 包（无需 git，用 urllib 下载 zip）
    _ensure_musk()
 
    import torchvision
    from timm.models import create_model
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    from musk import utils, modeling  # noqa — 触发 timm 模型注册
 
    # ── 1. 创建模型架构 ──────────────────────────────────────
    model = create_model("musk_large_patch16_384")
 
    # ── 2. 从 HF Hub 加载权重（官方方式）────────────────────
    utils.load_model_and_may_interpolate(
        "hf_hub:xiangjx/musk",
        model,
        "model|module",
        ""
    )
    model.to(device=device, dtype=torch.float16)
    model.eval()
 
    # ── 3. 官方图像预处理 ────────────────────────────────────
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
 
    # ── 4. encode_image（官方方式）──────────────────────────
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        """
        官方参数：
            with_head=False  : 不加投影头
            out_norm=True    : 输出归一化
            ms_aug=True      : 多尺度增强（适合线性探针/MIL）
            return_global=True: 只返回 [CLS] token
        返回 vision_cls，shape = [B, 2048]
        """
        with torch.inference_mode():
            vision_cls, _ = model(
                image=img_batch.to(device, dtype=torch.float16),
                with_head=False,
                out_norm=True,
                ms_aug=True,
                return_global=True,
            )
        return vision_cls.float()
 
    # ── 5. encode_text（官方方式）───────────────────────────
    def _encode_text(texts: List[str]) -> torch.Tensor:
        """
        使用 MUSK 自带的 XLM-RoBERTa tokenizer
        tokenizer 路径：musk/models/tokenizer.spm（包内自带）
        返回 text_cls，shape = [B, 2048]
        """
        import os
        from musk import utils as musk_utils
        from sentencepiece import SentencePieceProcessor
        # 找到 tokenizer.spm 的绝对路径（在 musk 包目录下）
        import musk as musk_pkg
        pkg_dir = os.path.dirname(musk_pkg.__file__)
        spm_path = os.path.join(pkg_dir, "models", "tokenizer.spm")
 
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer(spm_path)
 
        txt_ids, pad = musk_utils.xlm_tokenizer(texts, tokenizer, max_len=100)
        txt_ids = txt_ids.to(device)
        pad = pad.to(device)
 
        with torch.inference_mode():
            _, text_cls = model(
                text_description=txt_ids,
                padding_mask=pad,
                with_head=False,
                out_norm=True,
                ms_aug=False,
                return_global=True,
            )
        return text_cls.float()
 
    return FMWrapper(
        name="musk",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=384,
        precision=torch.float16,
    )
 
 
# ============================================================
# TITAN (MahmoodLab/TITAN)
# ============================================================

@register_fm("titan")
def load_titan(device: str = "cuda:0") -> FMWrapper:
    """加载 TITAN 模型"""
    repo_id = "MahmoodLab/TITAN"
    
    processor = _load_processor(repo_id)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        inputs = processor(images=pil_img, text="",return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            inputs = {"pixel_values": img_batch}
            outputs = model(**inputs)
            
            if hasattr(outputs, 'image_embeds'):
                feat = outputs.image_embeds
            elif hasattr(outputs, 'last_hidden_state'):
                feat = outputs.last_hidden_state[:, 0]
            elif hasattr(outputs, 'pooler_output'):
                feat = outputs.pooler_output
            else:
                feat = outputs[0][:, 0]
            
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    return FMWrapper(
        name="titan",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )


# ============================================================
# 公共入口
# ============================================================

def load_fm(name: str, device: str = "cuda:0", **kwargs) -> FMWrapper:
    """
    统一的对外接口
    用法：
        fm = load_fm("musk",         device="cuda:0")
        fm = load_fm("pathgen-clip", device="cuda:0")
        fm = load_fm("pathgen-clip", device="cuda:0", ckpt_path="/path/to/model.pt")
    """
    if name not in FM_REGISTRY:
        raise KeyError(f"Unknown FM name: {name}. Available: {list(FM_REGISTRY.keys())}")
    return FM_REGISTRY[name](device=device, **kwargs)


# ============================================================
# CONCH (MahmoodLab/CONCH) — Nature Medicine 2024
# ============================================================
#
# HF repo  : MahmoodLab/CONCH  (note: repo name is lowercase "conch")
# 状态     : ⚠️  Gated — 需登录 HF 并同意条款（机构邮箱）
# Paper    : "A visual-language foundation model for computational pathology"
#            Nature Medicine, 2024
# GitHub   : https://github.com/mahmoodlab/CONCH
# Arch     : ViT-B/16 (vision) + L12-E768-H12 (text)，CoCa 框架
# License  : CC-BY-NC-ND 4.0
#
# 官方加载方式：
#   pip install git+https://github.com/Mahmoodlab/CONCH.git
#   from conch.open_clip_custom import create_model_from_pretrained
#   model, preprocess = create_model_from_pretrained(
#       'conch_ViT-B-16', "hf_hub:MahmoodLab/conch",
#       hf_auth_token="<your_token>"
#   )
#
# encode_image 参数说明：
#   proj_contrast=False, normalize=False → 适合线性探针 / WSI 特征提取
#   proj_contrast=True,  normalize=True  → 适合 zero-shot 分类
#
# 前置条件：
#   1. 访问 https://huggingface.co/MahmoodLab/CONCH 同意条款（机构邮箱）
#   2. huggingface-cli login
#   3. pip install git+https://github.com/Mahmoodlab/CONCH.git
#      （或用 _ensure_conch() 自动安装，需要网络访问 GitHub）
# ============================================================
 
def _ensure_conch():
    """
    安装 CONCH 官方包
    官方安装: pip install git+https://github.com/Mahmoodlab/CONCH.git
    若无 git，改用 urllib 下载 zip 安装
    """
    import importlib
    try:
        importlib.import_module("conch")
        return  # 已安装
    except ImportError:
        pass
 
    import subprocess, sys, os, urllib.request, zipfile
    print("[CONCH] Installing CONCH package ...")
 
    zip_path = "/tmp/conch_install.zip"
    conch_dir = "/tmp/CONCH-main"
 
    if not os.path.exists(conch_dir):
        print("[CONCH] Downloading source from GitHub ...")
        urllib.request.urlretrieve(
            "https://github.com/mahmoodlab/CONCH/archive/refs/heads/main.zip",
            zip_path,
        )
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("/tmp")
 
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "-e", conch_dir
    ])
    print("[CONCH] ✅ CONCH package installed")
 
 
@register_fm("conch")
def load_conch(
    device: str = "cuda:0",
    ckpt_path: Optional[str] = None,
    hf_auth_token: Optional[str] = None,
    proj_contrast: bool = False,
    normalize: bool = False,
) -> FMWrapper:
    """
    加载 CONCH 模型（完全按照官方 README）
    HF repo : MahmoodLab/CONCH  (Gated，需登录 & 同意条款)
    Paper   : Nature Medicine 2024
    Arch    : ViT-B/16 (vision) + L12-E768-H12 (text)
    License : CC-BY-NC-ND 4.0
 
    参数
    ----
    ckpt_path      : 本地 pytorch_model.bin 路径（可选）
                     若为 None，从 HF Hub 自动下载（需已登录）
    hf_auth_token  : HuggingFace token（可选，已 login 则不需要）
    proj_contrast  : False → 适合线性探针 / WSI 特征提取（官方推荐）
                     True  → 适合 zero-shot 分类
    normalize      : False → 不归一化（适合线性探针）
                     True  → L2 归一化（适合 zero-shot）
 
    官方安装：
        pip install git+https://github.com/Mahmoodlab/CONCH.git
 
    前置条件：
        1. 访问 https://huggingface.co/MahmoodLab/CONCH 同意条款
        2. huggingface-cli login
    """
    import os
 
    # ── 1. 自动安装 conch 包 ──────────────────────────────────
    _ensure_conch()
    from conch.open_clip_custom import create_model_from_pretrained
 
    # ── 2. 加载模型（官方两种方式）───────────────────────────
    if ckpt_path is not None:
        # 方式一：本地文件（官方推荐离线方式）
        print(f"[CONCH] Loading from local checkpoint: {ckpt_path}")
        model, preprocess_fn = create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path=ckpt_path,
        )
    else:
        # 方式二：从 HF Hub 下载（需已登录 & 同意条款）
        print("[CONCH] Loading from HF Hub: MahmoodLab/conch")
        kwargs = {}
        if hf_auth_token:
            kwargs["hf_auth_token"] = hf_auth_token
        elif os.environ.get("HF_TOKEN"):
            kwargs["hf_auth_token"] = os.environ["HF_TOKEN"]
 
        model, preprocess_fn = create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            **kwargs,
        )
 
    model = model.to(device)
    model.eval()
 
    # ── 3. 保存 proj_contrast/normalize 供 encode 使用 ───────
    _proj_contrast = proj_contrast
    _normalize = normalize
 
    # ── 4. 图像预处理（由官方 create_model_from_pretrained 返回）
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return preprocess_fn(pil_img.convert("RGB"))
 
    # ── 5. encode_image（官方方式）──────────────────────────
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        """
        官方用法：
          proj_contrast=False, normalize=False
            → 线性探针 / WSI 特征提取（推荐）
          proj_contrast=True, normalize=True
            → zero-shot 分类
        """
        with torch.inference_mode():
            feat = model.encode_image(
                img_batch,
                proj_contrast=_proj_contrast,
                normalize=_normalize,
            )
        return feat
 
    # ── 6. encode_text（官方方式）───────────────────────────
    def _encode_text(texts: List[str]) -> torch.Tensor:
        """
        使用 CONCH 内置 tokenizer 编码文本
        """
        from conch.open_clip_custom import tokenize
        tokens = tokenize(texts=texts).to(device)
        with torch.inference_mode():
            feat = model.encode_text(tokens)
        return feat
 
    return FMWrapper(
        name="conch",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=True,
        encode_image=_encode_image,
        encode_text=_encode_text,
        image_size=448,
    )
# ============================================================
# BiomedCLIP (Microsoft - 公开可用)
# ============================================================

@register_fm("biomedclip")
def load_biomedclip(device: str = "cuda:0") -> FMWrapper:
    """加载 BiomedCLIP 模型（公开可用）"""
    repo_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    
    processor = _load_processor(repo_id)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        inputs = processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            inputs = {"pixel_values": img_batch}
            outputs = model(**inputs)
            
            if hasattr(outputs, 'image_embeds'):
                feat = outputs.image_embeds
            elif hasattr(outputs, 'last_hidden_state'):
                feat = outputs.last_hidden_state[:, 0]
            elif hasattr(outputs, 'pooler_output'):
                feat = outputs.pooler_output
            else:
                feat = outputs[0][:, 0]
            
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    def _encode_text(texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            if hasattr(outputs, 'text_embeds'):
                feat = outputs.text_embeds
            elif hasattr(outputs, 'last_hidden_state'):
                feat = outputs.last_hidden_state[:, 0]
            elif hasattr(outputs, 'pooler_output'):
                feat = outputs.pooler_output
            else:
                feat = outputs[0][:, 0]
            
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat

    return FMWrapper(
        name="biomedclip",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )

# ============================================================
# Patho-CLIP-L (WenchuanZhang/Patho-CLIP-L)
# ============================================================
#
# HF repo  : WenchuanZhang/Patho-CLIP-L
# 状态     : ⚠️  Gated — 需登录 HF 并同意条款（机构邮箱）
# Paper    : Patho-R1 (arXiv:2505.11404)
# Arch     : ViT-L/14，基于 OpenAI CLIP-L，OpenCLIP 框架
#            Stage I : PathGen-1.6M 对比预训练
#            Stage II: 3.5M 复合语料库联合训练
# License  : CC-BY-NC-ND 4.0
#
# 前置条件：
#   1. 访问 https://huggingface.co/WenchuanZhang/Patho-CLIP-L 同意条款
#   2. huggingface-cli login
# ============================================================
 
@register_fm("patho-clip")
def load_patho_clip(
    device: str = "cuda:0",
    ckpt_path: Optional[str] = None,
) -> FMWrapper:
    """
    加载 Patho-CLIP-L (ViT-L/14)
    HF repo : WenchuanZhang/Patho-CLIP-L  (Gated，需登录同意条款)
    Paper   : Patho-R1 (arXiv:2505.11404)
    Arch    : ViT-L/14，OpenCLIP 框架
    License : CC-BY-NC-ND 4.0
    依赖    : open_clip_torch  ← 自动安装
 
    权重获取（两种方式）：
      方式一（推荐）：从 HF Hub 自动下载（需登录）
        load_fm("patho-clip", device="cuda:0")
 
      方式二：手动下载后指定路径
        load_fm("patho-clip", device="cuda:0", ckpt_path="/path/to/patho_clip_l.pt")
    """
    _ensure_packages(open_clip="open_clip_torch")
    import open_clip
 
    if ckpt_path is not None:
        # 从本地 .pt 文件加载（ViT-L/14 架构）
        model, _, preprocess_fn = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=ckpt_path
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    else:
        # 从 HF Hub 下载 Patho-CLIP-L.pt，再用 ViT-L-14 架构加载
        import os
        from huggingface_hub import hf_hub_download
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
 
    # def _encode_text(texts: List[str]) -> torch.Tensor:
    #     tokens = tokenizer(texts).to(device)
    #     with torch.no_grad():
    #         feat = model.encode_text(tokens)
    #         feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
    #     return feat
 
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

# ============================================================
# HistoCLIP (marrlab/HistoGPT 项目)
# ============================================================
#
# HF 状态 : ⚠️  无独立 HF 页面
#            权重在 marr-peng-lab/histogpt（HistoGPT 项目）
# Paper   : "Generating dermatopathology reports from gigapixel WSIs"
#            Nature Communications 2025
# GitHub  : https://github.com/marrlab/HistoGPT
#
# Arch    : Slide-level CLIP（非 patch-level）
#            Vision : CTransPath（patch encoder）+ Perceiver Resampler（slide aggregator）
#            Text   : BioGPT（EOS token 作为 text embedding）
#            Loss   : CLIP contrastive（slide-level）
#
# ⚠️  重要说明：
#   HistoCLIP 是 slide-level 模型，输入是整张 WSI 的 patch 集合，
#   而不是单个 patch。与 PLIP/CONCH 等 patch-level CLIP 不同。
#   此处实现的是从 HistoGPT checkpoint 提取 HistoCLIP 组件。
#
# 权重文件（在 marr-peng-lab/histogpt）：
#   - histogpt-l-5k-pruned.pt  (13.1GB，推荐)
#   - histogpt-1b-5k-pruned.pth (3.46GB，轻量版)
# ============================================================
 
@register_fm("histoclip")
def load_histoclip(
    device: str = "cuda:0",
    ckpt_path: Optional[str] = None,
) -> FMWrapper:
    """
    加载 HistoCLIP（从 HistoGPT checkpoint 提取）
    HF repo : marr-peng-lab/histogpt  (公开，无需登录，Apache-2.0)
    Paper   : Nature Communications 2025
    GitHub  : https://github.com/marrlab/HistoGPT
    Arch    : CTransPath + Perceiver Resampler（slide-level）
    License : Apache-2.0
 
    参数
    ----
    ckpt_path : 本地权重路径。若为 None，自动从 HF 下载
                histogpt-1b-5k-pruned.pth（3.46GB，轻量版）
 
    ⚠️  注意：HistoCLIP 是 slide-level 模型
      - encode_image 输入: patch embeddings [N, D]（整张 slide 的所有 patches）
      - 不是直接输入单张 patch 图像
      - 如需 patch-level，请使用 plip / conch / biomedclip-v2 等
 
    依赖：open_clip_torch（自动安装）
    """
    import os
    from huggingface_hub import hf_hub_download
    from torchvision import transforms as T
 
    _ensure_packages(open_clip="open_clip_torch")
    import open_clip
 
    # ── 1. 下载权重（从 marr-peng-lab/histogpt）──────────────
    if ckpt_path is None:
        cache_dir = os.path.expanduser("~/.cache/histoclip")
        ckpt_path = os.path.join(cache_dir, "histogpt-1b-5k-pruned.pth")
        if not os.path.exists(ckpt_path):
            os.makedirs(cache_dir, exist_ok=True)
            print("[HistoCLIP] Downloading histogpt-1b-5k-pruned.pth from HF (3.46GB) ...")
            hf_hub_download(
                repo_id="marr-peng-lab/histogpt",
                filename="histogpt-1b-5k-pruned.pth",
                local_dir=cache_dir,
            )
            print(f"[HistoCLIP] ✅ Downloaded to {ckpt_path}")
 
    # ── 2. 加载 checkpoint ────────────────────────────────────
    print(f"[HistoCLIP] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
 
    # ── 3. 提取 visual encoder（CTransPath 部分）─────────────
    visual_state_dict = {
        k.replace("module.visual.", "").replace("visual.", ""): v
        for k, v in state_dict.items()
        if "visual" in k.lower()
    }
 
    # 用 ViT-B/16 承载视觉权重
    vision_model, _, preprocess_fn = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=None
    )
    if visual_state_dict:
        missing, _ = vision_model.visual.load_state_dict(visual_state_dict, strict=False)
        print(f"[HistoCLIP] Visual encoder loaded ({len(missing)} missing keys)")
    else:
        print("[HistoCLIP] ⚠️  No visual keys found, using random init")
 
    vision_model = vision_model.to(device).eval()
 
    # ── 4. 文本编码器（BioGPT）────────────────────────────────
    from transformers import BioGptTokenizer, BioGptModel
    _ensure_packages(sacremoses="sacremoses")
    print("[HistoCLIP] Loading BioGPT text encoder ...")
    bio_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
 
    # ── 5. 图像预处理 ──────────────────────────────────────────
    _transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
 
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        return _transform(pil_img.convert("RGB"))
 
    def _encode_image(img_batch: torch.Tensor) -> torch.Tensor:
        """提取 patch-level 视觉特征（CLS token）"""
        with torch.no_grad():
            feat = vision_model.encode_image(img_batch)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat
 
    def _encode_text(texts: List[str]) -> torch.Tensor:
        """用 BioGPT 编码文本，取 EOS token（官方方式）"""
        enc = bio_tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = bio_model(**enc)
            # 取最后一个有效 token（EOS token）
            seq_len = enc["attention_mask"].sum(dim=1) - 1  # [B]
            feat = out.last_hidden_state[
                torch.arange(len(texts)), seq_len
            ]  # [B, D]
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat
 
    return FMWrapper(
        name="histoclip",
        model=vision_model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )


# ============================================================
# MI-Zero (mahmoodlab/MI-Zero) — CVPR 2023
# ============================================================
#
# HF 状态 : ❌ 无 HuggingFace 页面
# GitHub  : https://github.com/mahmoodlab/MI-Zero
# Paper   : CVPR 2023 — Visual Language Pretrained Multiple Instance
#            Zero-Shot Transfer for Histopathology Images
# License : CC BY-NC-ND 4.0
#
# Arch    :
#   Vision  : CTransPath (custom Swin Transformer, 448×448)
#             需要自定义 timm_ctp 包（MI-Zero repo 提供）
#   Text    : BioClinicalBERT 或 PubMedBERT（HF transformers）
#
# 权重文件（需手动从 Google Drive 下载）：
#   bioclinicalbert 版 : ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt
#   pubmedbert 版      : ctranspath_448_pubmedbert/checkpoints/epoch_50.pt
#
# ⚠️  手动步骤：
#   1. 访问 https://github.com/mahmoodlab/MI-Zero
#   2. 从 README 中的 Google Drive 链接下载 checkpoint
#   3. 安装 timm_ctp（MI-Zero repo 提供）：
#      git clone https://github.com/mahmoodlab/MI-Zero
#      pip install ./MI-Zero/assets/timm_ctp.tar --no-deps
#   4. 通过 ckpt_path 参数传入权重路径
#
# Checkpoint 结构（epoch_50.pt）：
#   {
#     'epoch': 50,
#     'state_dict': {
#       'module.visual.xxx': ...,  # CTransPath visual encoder
#       'module.text.xxx': ...,    # Text projection layers
#       'module.logit_scale': ..., # Temperature parameter
#     }
#   }
# ============================================================
 
def _ensure_timm_ctp():
    """
    安装 timm_ctp（MI-Zero 所需的自定义 timm 版本）
    MI-Zero 官方安装方式:
        pip install ./assets/timm_ctp.tar --no-deps
    此处使用 urllib 从 MI-Zero repo 下载后安装
    """
    import importlib
    try:
        import timm_ctp  # noqa
        return  # 已安装
    except ImportError:
        pass
 
    import subprocess, sys, os, urllib.request, zipfile
    print("[MI-Zero] Installing timm_ctp ...")
 
    # 先下载 MI-Zero repo zip
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
    text_encoder: str = "bioclinicalbert",  # "bioclinicalbert" | "pubmedbert"
) -> FMWrapper:
    """
    加载 MI-Zero 模型（CVPR 2023，MahmoodLab）
    GitHub  : https://github.com/mahmoodlab/MI-Zero
    HF 状态 : ❌ 无 HF 页面，需手动下载权重
    Arch    : CTransPath (448×448) + BioClinicalBERT/PubMedBERT
    License : CC BY-NC-ND 4.0
 
    参数
    ----
    ckpt_path    : 本地 epoch_50.pt 路径（必须手动下载）
    text_encoder : "bioclinicalbert" (默认) 或 "pubmedbert"
 
    手动下载权重：
      1. 访问 https://github.com/mahmoodlab/MI-Zero
      2. 在 README 找到 Google Drive 链接下载：
         - bioclinicalbert: ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt
         - pubmedbert     : ctranspath_448_pubmedbert/checkpoints/epoch_50.pt
      3. load_fm("mi-zero", ckpt_path="/path/to/epoch_50.pt")
 
    依赖：
      - timm_ctp（自动安装，需网络）
      - transformers（BioClinicalBERT/PubMedBERT）
    """
    from torchvision import transforms as T
    from transformers import AutoTokenizer, AutoModel as HFAutoModel
 
    if ckpt_path is None:
        raise ValueError(
            "\n[MI-Zero] 无 HuggingFace 页面，需手动下载权重。\n"
            "请访问 https://github.com/mahmoodlab/MI-Zero\n"
            "下载 Google Drive 链接中的 checkpoint：\n"
            "  BioClinicalBERT: ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt\n"
            "  PubMedBERT     : ctranspath_448_pubmedbert/checkpoints/epoch_50.pt\n"
            "然后调用：\n"
            "  load_fm('mi-zero', ckpt_path='/path/to/epoch_50.pt')"
        )
 
    # ── 1. 安装 timm_ctp（CTransPath 所需）──────────────────
    _ensure_timm_ctp()
 
    # ── 2. 加载 checkpoint ────────────────────────────────────
    print(f"[MI-Zero] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
 
    # checkpoint 结构: {'epoch': N, 'state_dict': {...}}
    state_dict = ckpt.get("state_dict", ckpt)
 
    # 去掉 'module.' 前缀（DDP 训练保存的）
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "", 1)
        new_state_dict[new_key] = v
    state_dict = new_state_dict
 
    # ── 3. 构建 CTransPath 视觉编码器 ─────────────────────────
    # CTransPath 使用 timm_ctp 中的自定义 Swin Transformer
    try:
        import timm_ctp
        visual_model = timm_ctp.create_model(
            "swin_tiny_patch4_window7_224",
            embed_layer=timm_ctp.models.swin_transformer.ConvStem,
            pretrained=False,
            num_classes=0,
            img_size=448,  # CTransPath 使用 448x448 输入
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
 
    # 提取 visual 权重（前缀为 'visual.'）
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
 
    # ── 4. 文本编码器（从 HF 加载）────────────────────────────
    if text_encoder == "bioclinicalbert":
        text_model_id = "emilyalsentzer/Bio_ClinicalBERT"
    elif text_encoder == "pubmedbert":
        text_model_id = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    else:
        raise ValueError(f"text_encoder must be 'bioclinicalbert' or 'pubmedbert', got: {text_encoder}")
 
    print(f"[MI-Zero] Loading text encoder: {text_model_id}")
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_model = HFAutoModel.from_pretrained(text_model_id).to(device).eval()
 
    # ── 5. 图像预处理（官方 448×448）─────────────────────────
    # 来自 MI-Zero 官方 extract_embeddings.py
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
        """
        提取 patch-level 视觉特征
        CTransPath 输出: [B, D]（Swin 全局平均池化后）
        """
        with torch.no_grad():
            # Swin Transformer forward_features 返回 [B, H, W, C]
            feat = visual_model.forward_features(img_batch)
            # 全局平均池化
            if feat.dim() == 4:
                feat = feat.mean(dim=[1, 2])  # [B, C]
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)       # [B, C]
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat
 
    def _encode_text(texts: List[str]) -> torch.Tensor:
        """
        用 BioClinicalBERT/PubMedBERT 编码文本
        取 [CLS] token 作为句子表示
        """
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
            feat = out.last_hidden_state[:, 0]  # [CLS] token
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

# ============================================================
# CPath-CLIP
# ============================================================

@register_fm("cpath-clip")
def load_cpath_clip(
    device: str = "cuda:0",
    ckpt_path: Optional[str] = None,
) -> FMWrapper:
    _ensure_packages(open_clip="open_clip_torch")
    import open_clip
    
    if ckpt_path is None:
        raise ValueError(
            "\n[CPath-CLIP] 需要本地权重文件。\n"
            "请下载 cpath_clip_delta.pt 并通过 ckpt_path 传入：\n"
            "  load_fm('cpath-clip', ckpt_path='/path/to/cpath_clip_delta.pt')"
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
        name="cpath-clip",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=False,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )

    attach_text_adapter(fm, tokenizer=tokenizer, device=device)
    return fm

# ============================================================
# QuiltNet-B-16
# https://huggingface.co/wisdomik/QuiltNet-B-16
# ============================================================
@register_fm("quiltnet")
def load_quiltnet(device: str = "cuda:0") -> FMWrapper:
    _ensure_packages(open_clip="open_clip_torch")
    import open_clip

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

    fm = FMWrapper(
        name="quiltnet",
        model=model,
        preprocess=_preprocess,
        has_text_encoder=True,
        encode_image=_encode_image,
        encode_text=None,
        image_size=224,
    )
    return fm

if __name__ == "__main__":
    print("Available models:", list(FM_REGISTRY.keys()))

