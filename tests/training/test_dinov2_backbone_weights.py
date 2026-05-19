# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for DINOv2 backbone-only checkpoint loading."""

from __future__ import annotations

import torch
from torch import nn

from rfdetr.models.weights import load_dinov2_backbone_weights


class _LayerScale(nn.Module):
    """Minimal layer-scale module with a DINOv2-compatible state key."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.zeros(dim))


class _Block(nn.Module):
    """Minimal DINOv2 block exposing RF-DETR target state-dict keys."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.Module()
        self.attention.attention = nn.Module()
        self.attention.attention.query = nn.Linear(dim, dim)
        self.attention.attention.key = nn.Linear(dim, dim)
        self.attention.attention.value = nn.Linear(dim, dim)
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(dim, dim)
        self.layer_scale1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 4)
        self.mlp.fc2 = nn.Linear(dim * 4, dim)
        self.layer_scale2 = _LayerScale(dim)


class _FakeEncoder(nn.Module):
    """Minimal RF-DETR DINOv2 encoder for loader tests."""

    def __init__(self, dim: int = 4, depth: int = 2, patch_size: int = 14) -> None:
        super().__init__()
        self.config = type(
            "Cfg",
            (),
            {
                "hidden_size": dim,
                "num_hidden_layers": depth,
                "num_attention_heads": 2,
                "mlp_ratio": 4,
                "patch_size": patch_size,
                "image_size": 28,
                "num_register_tokens": 1,
            },
        )()
        self.embeddings = nn.Module()
        self.embeddings.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.embeddings.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.embeddings.register_tokens = nn.Parameter(torch.zeros(1, 1, dim))
        self.embeddings.position_embeddings = nn.Parameter(torch.zeros(1, 5, dim))
        self.embeddings.patch_embeddings = nn.Module()
        self.embeddings.patch_embeddings.projection = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Block(dim) for _ in range(depth)])
        self.layernorm = nn.LayerNorm(dim)


class _FakeRFDETR(nn.Module):
    """Minimal RF-DETR-like wrapper exposing ``backbone[0].encoder``."""

    def __init__(self) -> None:
        super().__init__()
        backbone = nn.Module()
        backbone.encoder = _FakeEncoder()
        self.backbone = nn.Sequential(backbone)
        self.transformer = nn.Linear(4, 4)
        self.class_embed = nn.Linear(4, 3)
        self.bbox_embed = nn.Linear(4, 4)


def _original_dinov2_checkpoint_from_target(target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create original DINOv2-style keys from fake RF-DETR target keys."""
    state = {
        "cls_token": torch.randn_like(target["embeddings.cls_token"]),
        "mask_token": torch.randn_like(target["embeddings.mask_token"]),
        "register_tokens": torch.randn_like(target["embeddings.register_tokens"]),
        "pos_embed": torch.randn_like(target["embeddings.position_embeddings"]),
        "patch_embed.proj.weight": torch.randn_like(target["embeddings.patch_embeddings.projection.weight"]),
        "patch_embed.proj.bias": torch.randn_like(target["embeddings.patch_embeddings.projection.bias"]),
        "norm.weight": torch.randn_like(target["layernorm.weight"]),
        "norm.bias": torch.randn_like(target["layernorm.bias"]),
    }
    for layer_idx in range(2):
        prefix = f"encoder.layer.{layer_idx}"
        original = f"blocks.{layer_idx}"
        q = torch.randn_like(target[f"{prefix}.attention.attention.query.weight"])
        k = torch.randn_like(target[f"{prefix}.attention.attention.key.weight"])
        v = torch.randn_like(target[f"{prefix}.attention.attention.value.weight"])
        state[f"{original}.attn.qkv.weight"] = torch.cat([q, k, v], dim=0)
        qb = torch.randn_like(target[f"{prefix}.attention.attention.query.bias"])
        kb = torch.randn_like(target[f"{prefix}.attention.attention.key.bias"])
        vb = torch.randn_like(target[f"{prefix}.attention.attention.value.bias"])
        state[f"{original}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
        mapping = {
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "attn.proj.weight": "attention.output.dense.weight",
            "attn.proj.bias": "attention.output.dense.bias",
            "ls1.gamma": "layer_scale1.lambda1",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
            "mlp.fc1.weight": "mlp.fc1.weight",
            "mlp.fc1.bias": "mlp.fc1.bias",
            "mlp.fc2.weight": "mlp.fc2.weight",
            "mlp.fc2.bias": "mlp.fc2.bias",
            "ls2.gamma": "layer_scale2.lambda1",
        }
        for source_suffix, target_suffix in mapping.items():
            state[f"{original}.{source_suffix}"] = torch.randn_like(target[f"{prefix}.{target_suffix}"])
    return state


def test_load_dinov2_backbone_weights_maps_original_qkv_keys(tmp_path) -> None:
    """Original DINOv2 qkv keys are split and loaded into RF-DETR query/key/value keys."""
    model = _FakeRFDETR()
    target = model.backbone[0].encoder.state_dict()
    checkpoint_state = _original_dinov2_checkpoint_from_target(target)
    checkpoint_path = tmp_path / "dinov2_original.pth"
    torch.save(checkpoint_state, checkpoint_path)

    report = load_dinov2_backbone_weights(model, str(checkpoint_path))

    assert report.loaded_blocks == 2
    assert report.total_blocks == 2
    assert report.missing_keys == []
    assert report.shape_mismatched_keys == {}
    assert report.module_ratios["patch_embed"] == 1.0
    assert report.module_ratios["transformer_blocks"] == 1.0
    loaded = model.backbone[0].encoder.state_dict()
    q, k, v = checkpoint_state["blocks.0.attn.qkv.weight"].chunk(3, dim=0)
    assert torch.equal(loaded["encoder.layer.0.attention.attention.query.weight"], q)
    assert torch.equal(loaded["encoder.layer.0.attention.attention.key.weight"], k)
    assert torch.equal(loaded["encoder.layer.0.attention.attention.value.weight"], v)


def test_load_dinov2_backbone_weights_raises_on_shape_mismatch(tmp_path) -> None:
    """Shape mismatches are reported and raise instead of being silently skipped."""
    model = _FakeRFDETR()
    target = model.backbone[0].encoder.state_dict()
    checkpoint_state = _original_dinov2_checkpoint_from_target(target)
    checkpoint_state["patch_embed.proj.weight"] = torch.randn(99, 3, 14, 14)
    checkpoint_path = tmp_path / "bad_dinov2.pth"
    torch.save(checkpoint_state, checkpoint_path)

    try:
        load_dinov2_backbone_weights(model, str(checkpoint_path))
    except RuntimeError as exc:
        assert "shape" in str(exc).lower()
        assert "embeddings.patch_embeddings.projection.weight" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch to raise RuntimeError.")
