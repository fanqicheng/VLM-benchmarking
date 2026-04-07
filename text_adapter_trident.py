#!/usr/bin/env python3
"""
text_adapter.py

A dedicated text adapter layer for pathology VLMs.
Goal:
    - unify fm.encode_text(texts) across heterogeneous model families
    - avoid forcing all models into one tokenizer/one API

Supported:
    - conch
    - plip
    - keep
    - biomedclip-v2
    - pathgen-clip
    - patho-clip
    - histoclip   (best-effort, not official zero-shot API)

Notes:
    - Some models are true OpenCLIP-style
    - Some require HF tokenizers
    - Some are custom / partial support only
"""

from __future__ import annotations

from typing import Callable, List, Optional
import inspect
import torch
import torch.nn.functional as F


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _batched(texts: List[str], batch_size: int = 32):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


def _move_batch_to_device(batch, device: str):
    if isinstance(batch, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _extract_text_features_from_output(outputs) -> torch.Tensor:
    """
    Generic extractor for HF/custom outputs.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs

    for attr in ["text_embeds", "pooler_output", "last_hidden_state"]:
        if hasattr(outputs, attr):
            feat = getattr(outputs, attr)
            if attr == "last_hidden_state" and feat.dim() == 3:
                feat = feat[:, 0]
            return feat

    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        feat = outputs[0]
        if isinstance(feat, torch.Tensor) and feat.dim() == 3:
            feat = feat[:, 0]
        return feat

    raise RuntimeError(f"Cannot extract text features from output type: {type(outputs)}")


# ---------------------------------------------------------------------
# Generic builders
# ---------------------------------------------------------------------

def build_openclip_text_adapter(
    model,
    tokenizer,
    device: str,
    *,
    batch_size: int = 32,
    context_length: Optional[int] = None,
) -> Callable[[List[str]], torch.Tensor]:
    """
    OpenCLIP-style adapter:
        tokens = tokenizer(texts)
        feat = model.encode_text(tokens)
    Works for:
        - PLIP
        - PathGen-CLIP
        - Patho-CLIP (if loaded via OpenCLIP-style weights)
    """
    @torch.inference_mode()
    def encode_text(texts: List[str]) -> torch.Tensor:
        outs = []
        for chunk in _batched(texts, batch_size=batch_size):
            # tokenizer signatures differ slightly across repos
            if context_length is not None:
                try:
                    tokens = tokenizer(chunk, context_length=context_length)
                except TypeError:
                    tokens = tokenizer(chunk)
            else:
                tokens = tokenizer(chunk)

            tokens = _move_batch_to_device(tokens, device)

            feat = model.encode_text(tokens)
            feat = _normalize(feat)
            outs.append(feat)

        return torch.cat(outs, dim=0)

    return encode_text


def build_hf_tokenizer_clip_encode_text_adapter(
    model,
    tokenizer,
    device: str,
    *,
    batch_size: int = 32,
    max_length: int = 256,
) -> Callable[[List[str]], torch.Tensor]:
    """
    HF-tokenizer + model.encode_text(inputs)
    Works for models where:
        - tokenizer is HF tokenizer
        - model.encode_text accepts a dict / BatchEncoding
    """
    @torch.inference_mode()
    def encode_text(texts: List[str]) -> torch.Tensor:
        outs = []
        for chunk in _batched(texts, batch_size=batch_size):
            inputs = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = _move_batch_to_device(inputs, device)

            feat = model.encode_text(inputs)
            feat = _normalize(feat)
            outs.append(feat)

        return torch.cat(outs, dim=0)

    return encode_text


def build_hf_forward_text_adapter(
    model,
    tokenizer,
    device: str,
    *,
    batch_size: int = 32,
    max_length: int = 256,
    prefer_get_text_features: bool = True,
) -> Callable[[List[str]], torch.Tensor]:
    """
    HF model adapter:
        - tokenizer(...)
        - model.get_text_features(**inputs) OR model(**inputs)
    Use this if encode_text is not reliable.
    """
    @torch.inference_mode()
    def encode_text(texts: List[str]) -> torch.Tensor:
        outs = []
        for chunk in _batched(texts, batch_size=batch_size):
            inputs = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = _move_batch_to_device(inputs, device)

            if prefer_get_text_features and hasattr(model, "get_text_features"):
                feat = model.get_text_features(**inputs)
            else:
                outputs = model(**inputs)
                feat = _extract_text_features_from_output(outputs)

            feat = _normalize(feat)
            outs.append(feat)

        return torch.cat(outs, dim=0)

    return encode_text


def build_biogpt_text_adapter(
    biogpt_model,
    biogpt_tokenizer,
    device: str,
    *,
    batch_size: int = 16,
    max_length: int = 512,
) -> Callable[[List[str]], torch.Tensor]:
    """
    BioGPT-style text adapter:
        uses last valid token as sentence embedding.
    Best-effort for HistoCLIP / HistoGPT-style language side.
    """
    @torch.inference_mode()
    def encode_text(texts: List[str]) -> torch.Tensor:
        outs = []
        for chunk in _batched(texts, batch_size=batch_size):
            enc = biogpt_tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = _move_batch_to_device(enc, device)

            outputs = biogpt_model(**enc)
            hidden = outputs.last_hidden_state  # [B, L, D]
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            feat = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
            feat = _normalize(feat)
            outs.append(feat)

        return torch.cat(outs, dim=0)

    return encode_text


# ---------------------------------------------------------------------
# Registry-facing entry point
# ---------------------------------------------------------------------

def attach_text_adapter(
    fm,
    *,
    tokenizer=None,
    hf_text_model=None,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> None:
    """
    Mutates fm in-place:
        - fm.encode_text = callable
        - fm.has_text_encoder = True

    Expected:
        fm.name identifies the model family.
    """
    if device is None:
        # best effort device inference
        try:
            device = next(fm.model.parameters()).device.type
            if device == "cuda":
                idx = next(fm.model.parameters()).device.index
                device = f"cuda:{idx}" if idx is not None else "cuda"
        except Exception:
            device = "cpu"

    name = fm.name.lower()

    # -------------------------------------------------------------
    # CONCH
    # Official usage relies on conch.open_clip_custom and encode_text.
    # The model supports text encoding according to official README. 
    # -------------------------------------------------------------
    if name == "conch":
        if tokenizer is None:
            try:
                from conch.open_clip_custom import get_tokenizer
                tokenizer = get_tokenizer()
            except Exception as e:
                raise RuntimeError(
                    "CONCH tokenizer not provided and automatic import failed. "
                    "Install CONCH and/or pass tokenizer from loader."
                ) from e

        fm.encode_text = build_openclip_text_adapter(
            fm.model, tokenizer, device, batch_size=batch_size
        )
        fm.has_text_encoder = True
        return

    # -------------------------------------------------------------
    # PLIP / PathGen-CLIP / Patho-CLIP
    # These are OpenCLIP-style usage patterns in official examples / model pages.
    # -------------------------------------------------------------
    if name in {"plip", "pathgen-clip", "patho-clip"}:
        if tokenizer is None:
            raise RuntimeError(f"{fm.name}: tokenizer must be provided from loader.")
        fm.encode_text = build_openclip_text_adapter(
            fm.model, tokenizer, device, batch_size=batch_size
        )
        fm.has_text_encoder = True
        return

    # -------------------------------------------------------------
    # KEEP
    # KEEP uses HF tokenizer + model.encode_text(dict_inputs) in your current loader.
    # -------------------------------------------------------------
    if name == "keep":
        if tokenizer is None:
            raise RuntimeError("KEEP: tokenizer must be provided from loader.")
        fm.encode_text = build_hf_tokenizer_clip_encode_text_adapter(
            fm.model, tokenizer, device, batch_size=batch_size, max_length=256
        )
        fm.has_text_encoder = True
        return

    # -------------------------------------------------------------
    # BioMedCLIP v2
    # Use HF tokenizer + get_text_features / forward, not generic open_clip tokenizer.
    # -------------------------------------------------------------
    if name == "biomedclip-v2":
        if tokenizer is None:
            raise RuntimeError("biomedclip-v2: tokenizer must be provided from loader.")

        # ❗BiomedCLIP = OpenCLIP family
        fm.encode_text = build_openclip_text_adapter(
            fm.model, tokenizer, device, batch_size=batch_size, context_length=77
        )

        fm.has_text_encoder = True
        return

    # -------------------------------------------------------------
    # HistoCLIP / HistoGPT
    # Public repo emphasizes BioGPT language module and generated reports;
    # public zero-shot text encoder tooling is incomplete.
    # This is a best-effort adapter, not guaranteed official retrieval behavior.
    # -------------------------------------------------------------
    if name == "histoclip":
        if hf_text_model is None or tokenizer is None:
            raise RuntimeError(
                "histoclip: requires both BioGPT tokenizer and BioGPT model "
                "from the loader for best-effort text embeddings."
            )
        fm.encode_text = build_biogpt_text_adapter(
            hf_text_model, tokenizer, device, batch_size=16, max_length=512
        )
        fm.has_text_encoder = True
        return

    raise KeyError(f"No text adapter registered for model name: {fm.name}")