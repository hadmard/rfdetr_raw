# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared weight-loading and LoRA application utilities.

Provides the canonical implementations of pretrained checkpoint loading and
LoRA adapter injection, used by both the L1 inference facade (``rfdetr.detr``)
and the L2 LightningModule (``rfdetr.training.module_model``).

The weight-loading logic is taken from ``RFDETRModelModule._load_pretrain_weights``
in ``module_model.py`` (more complete: Pydantic-aware user-override detection,
auto-alignment for fine-tuned checkpoints) and augmented with class-name
extraction from ``detr.py:_load_pretrain_weights_into``.
"""

from __future__ import annotations

import functools
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.utilities.decorators import deprecated
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.state_dict import _ckpt_args_get, validate_checkpoint_compatibility

logger = get_logger()

__all__ = [
    "DinoV2BackboneLoadReport",
    "apply_lora",
    "interpolate_position_embeddings",
    "load_dinov2_backbone_weights",
    "load_pretrain_weights",
]

_PE_KEY_SUFFIX = "embeddings.position_embeddings"

# Query-related parameters that LWDETR packs as nn.Embedding(num_queries * group_detr, ...).
# Any new query parameter packed the same way must be added here.
_QUERY_PARAM_SUFFIXES: tuple[str, ...] = ("refpoint_embed.weight", "query_feat.weight")


@dataclass
class DinoV2BackboneLoadReport:
    """Structured report for DINOv2 backbone-only checkpoint loading.

    Attributes:
        checkpoint_path: Path that was loaded.
        source_key_count: Number of tensor keys found in the checkpoint state.
        loaded_keys: Target backbone keys copied into the RF-DETR model.
        missing_keys: Target backbone keys not provided by the source checkpoint.
        unexpected_keys: Source tensor keys not consumed by the mapper.
        shape_mismatched_keys: Shape mismatch descriptions keyed by target key.
        skipped_keys: Source keys skipped for a known reason.
        total_model_params: Total parameter count for the full RF-DETR model.
        trainable_params: Trainable parameter count for the full RF-DETR model.
        backbone_params: Total parameter count in ``model.backbone``.
        loaded_backbone_params: Number of backbone parameter elements loaded.
        module_ratios: Per-module loaded parameter ratios.
        loaded_blocks: Number of transformer blocks with at least one loaded key.
        total_blocks: Number of RF-DETR backbone transformer blocks.
    """

    checkpoint_path: str
    source_key_count: int
    loaded_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatched_keys: dict[str, str]
    skipped_keys: dict[str, str]
    total_model_params: int
    trainable_params: int
    backbone_params: int
    loaded_backbone_params: int
    module_ratios: dict[str, float]
    loaded_blocks: int
    total_blocks: int

    @property
    def loaded_backbone_ratio(self) -> float:
        """Return loaded backbone parameter ratio."""
        if self.backbone_params == 0:
            return 0.0
        return self.loaded_backbone_params / self.backbone_params

    @property
    def unexpected_missing_backbone_keys(self) -> list[str]:
        """Return missing target backbone keys that are not intentionally excluded."""
        return self.missing_keys

    def to_log_lines(self) -> list[str]:
        """Return human-readable diagnostic lines."""
        lines = [
            "DINOv2 backbone-only load report:",
            f"  checkpoint_path: {self.checkpoint_path}",
            f"  source_key_count: {self.source_key_count}",
            f"  total_model_params: {self.total_model_params}",
            f"  trainable_params: {self.trainable_params}",
            f"  backbone_params: {self.backbone_params}",
            f"  loaded_backbone_params: {self.loaded_backbone_params}",
            f"  loaded_backbone_ratio: {self.loaded_backbone_ratio:.6f}",
            f"  loaded_keys_count: {len(self.loaded_keys)}",
            f"  missing_keys_count: {len(self.missing_keys)}",
            f"  unexpected_keys_count: {len(self.unexpected_keys)}",
            f"  shape_mismatched_keys_count: {len(self.shape_mismatched_keys)}",
            f"  skipped_keys_count: {len(self.skipped_keys)}",
            f"  transformer_blocks_loaded: {self.loaded_blocks}/{self.total_blocks}",
        ]
        for name, ratio in sorted(self.module_ratios.items()):
            lines.append(f"  {name}_loaded_ratio: {ratio:.6f}")
        if self.missing_keys:
            lines.append(f"  unexpected_missing_backbone_keys: {self.missing_keys}")
        if self.unexpected_keys:
            lines.append(f"  unexpected_keys: {self.unexpected_keys}")
        if self.shape_mismatched_keys:
            lines.append(f"  shape_mismatched_keys: {self.shape_mismatched_keys}")
        if self.skipped_keys:
            lines.append(f"  skipped_keys: {self.skipped_keys}")
        lines.extend(
            [
                "  expected_missing_non_backbone_keys: all decoder/query/projector/class_head/box_head keys",
                "  decoder_loaded: no",
                "  class_head_loaded: no",
                "  box_head_loaded: no",
            ]
        )
        return lines


def _slice_query_param_per_group(
    tensor: torch.Tensor,
    ckpt_num_queries: int,
    ckpt_group_detr: int,
    target_num_queries: int,
    target_group_detr: int,
) -> torch.Tensor:
    """Slice a ``refpoint_embed`` / ``query_feat`` weight preserving per-group structure.

    ``LWDETR`` packs query embeddings as ``nn.Embedding(num_queries * group_detr, ...)``
    where group ``g`` occupies the contiguous slot range
    ``[g * num_queries, (g + 1) * num_queries)`` (see ``LWDETR.__init__`` and
    ``LWDETR.forward`` in ``models/lwdetr.py``).  When ``num_queries`` decreases
    and ``group_detr > 1``, a flat ``tensor[: target_num_queries * target_group_detr]``
    slice silently scrambles groups: the tail of group 0 winds up in what should
    be group 1's slots, and so on.  At inference only group 0 is read so the
    bug is invisible, but for training-resume it corrupts groups 1+.

    This helper does the right thing per group:

    * ``num_queries`` decrease (``target_num_queries < ckpt_num_queries``) →
      keep the first ``target_num_queries`` slots of each retained group.
    * ``group_detr`` decrease (``target_group_detr < ckpt_group_detr``) →
      drop tail groups; retained groups stay pretrained.
    * Either dimension expands, or one shrinks while the other expands →
      return whatever per-group sub-tensor can be built (``min(target, ckpt)``
      along each axis). The result has fewer rows than the model expects, so
      ``load_state_dict`` will raise a shape mismatch immediately.

    When the tensor's flat length disagrees with
    ``ckpt_num_queries * ckpt_group_detr`` (corrupt or unexpected checkpoint
    shape), fall back to the legacy flat slice so loading continues with the
    same behavior the codebase had before this fix.

    Args:
        tensor: The checkpoint tensor for ``refpoint_embed.weight`` or
            ``query_feat.weight``.
        ckpt_num_queries: ``num_queries`` recorded in the checkpoint's training args.
        ckpt_group_detr: ``group_detr`` recorded in the checkpoint's training args.
        target_num_queries: ``num_queries`` configured for the model.
        target_group_detr: ``group_detr`` configured for the model.

    Returns:
        A tensor whose layout matches the model's configured packing for the
        decrease-or-equal cases, or a per-group sub-tensor built from
        ``min(target, ckpt)`` along each axis for the expansion case (which
        ``load_state_dict`` will then reject on shape mismatch).

    Raises:
        ValueError: If any of ``ckpt_num_queries``, ``ckpt_group_detr``,
            ``target_num_queries``, or ``target_group_detr`` is ≤ 0.
    """
    if ckpt_num_queries <= 0 or ckpt_group_detr <= 0 or target_num_queries <= 0 or target_group_detr <= 0:
        raise ValueError(
            f"_slice_query_param_per_group: all dimension args must be positive; "
            f"got ckpt_num_queries={ckpt_num_queries}, ckpt_group_detr={ckpt_group_detr}, "
            f"target_num_queries={target_num_queries}, target_group_detr={target_group_detr}."
        )

    expected_total = ckpt_num_queries * ckpt_group_detr
    if tensor.shape[0] != expected_total:
        # Args inconsistent with tensor shape — fall back to legacy flat slice.
        logger.warning(
            "_slice_query_param_per_group: checkpoint args claim %d × %d = %d rows "
            "but tensor has %d rows; falling back to flat slice. Per-group structure "
            "may be scrambled if group_detr > 1.",
            ckpt_num_queries,
            ckpt_group_detr,
            expected_total,
            tensor.shape[0],
        )
        return tensor[: target_num_queries * target_group_detr]

    if target_num_queries == ckpt_num_queries and target_group_detr == ckpt_group_detr:
        return tensor

    keep_groups = min(target_group_detr, ckpt_group_detr)
    keep_per_group = min(target_num_queries, ckpt_num_queries)
    pieces = [tensor[g * ckpt_num_queries : g * ckpt_num_queries + keep_per_group] for g in range(keep_groups)]
    return torch.cat(pieces, dim=0)


def _filter_intentional_keys(keys: list[str]) -> list[str]:
    """Return *keys* with intentional-reinit/trim entries removed.

    Matching is boundary-aware: a pattern matches a key when the pattern
    appears at the start of the key or immediately after a module separator
    (``.``).  This prevents substring collisions where a pattern like
    ``"class_embed."`` would inadvertently match a key belonging to an
    unrelated module (e.g. ``"class_embed_projection.weight"`` is safe
    because ``class_embed_projection.`` ≠ ``class_embed.``, but using a
    plain ``in`` check against longer ambiguous strings is fragile by
    design).
    """
    # Substrings identifying state-dict keys that ``load_pretrain_weights`` is
    # *expected* to have to reconcile (head reinitialisation and per-group query
    # trimming).  Keys matching any of these are filtered from the partial-load
    # warning so it only fires on *unexpected* mismatches that indicate a real
    # config / checkpoint incompatibility.
    intentional_patterns: tuple[str, ...] = (
        "class_embed.",
        "bbox_embed.",
        *_QUERY_PARAM_SUFFIXES,
        "enc_out_class_embed.",
        "enc_out_bbox_embed.",
    )

    def _is_intentional(key: str) -> bool:
        return any(key.startswith(pat) or f".{pat}" in key for pat in intentional_patterns)

    return [k for k in keys if not _is_intentional(k)]


def _warn_on_partial_load(incompatible: Any, pretrain_weights_path: str) -> None:
    """Emit a ``logger.warning`` when ``load_state_dict`` left non-trivial gaps.

    ``load_state_dict(strict=False)`` silently ignores keys that the model has
    but the checkpoint does not (``missing_keys``) and keys present in the
    checkpoint but absent from the model (``unexpected_keys``).  When this
    happens for parameters outside the head / query embeddings — which the
    loader intentionally reinitialises or trims — the corresponding model
    weights were left at their random initial values and the user is silently
    getting a much weaker model.

    This helper surfaces that condition with a single, actionable warning.
    Same-key shape mismatches do not reach this function — they raise
    :class:`RuntimeError` directly from ``load_state_dict`` and are therefore
    impossible to miss.

    Args:
        incompatible: The ``_IncompatibleKeys`` namedtuple returned by
            :meth:`torch.nn.Module.load_state_dict`.
        pretrain_weights_path: Path to the checkpoint that was loaded; included
            in the warning so the user can identify which load partially
            succeeded.
    """
    missing_keys_raw = getattr(incompatible, "missing_keys", None)
    unexpected_keys_raw = getattr(incompatible, "unexpected_keys", None)
    try:
        missing_keys = [str(k) for k in missing_keys_raw] if missing_keys_raw else []
        unexpected_keys = [str(k) for k in unexpected_keys_raw] if unexpected_keys_raw else []
    except TypeError:
        # Result wasn't iterable (e.g. a MagicMock in unit tests) — quietly skip.
        return
    missing = _filter_intentional_keys(missing_keys)
    unexpected = _filter_intentional_keys(unexpected_keys)
    if not missing and not unexpected:
        return

    parts: list[str] = []
    if missing:
        sample = ", ".join(missing[:5])
        if len(missing) > 5:
            sample += ", ..."
        parts.append(f"{len(missing)} model parameter(s) not in checkpoint (left at random init): [{sample}]")
    if unexpected:
        sample = ", ".join(unexpected[:5])
        if len(unexpected) > 5:
            sample += ", ..."
        parts.append(f"{len(unexpected)} checkpoint key(s) not consumed by model: [{sample}]")

    logger.warning(
        "Pretrained weights at %r loaded only partially — this typically produces "
        "lower accuracy. %s. Check that the model configuration (encoder, hidden_dim, "
        "out_feature_indexes, projector_scale, ...) matches the architecture the "
        "checkpoint was trained with.",
        pretrain_weights_path,
        " ".join(parts),
    )


def interpolate_position_embeddings(
    checkpoint_state: dict,
    pe_size: int,
) -> None:
    """Interpolate DINOv2 positional embeddings in *checkpoint_state* to match *pe_size*.

    When the model is configured with a custom ``resolution`` that differs from the
    checkpoint's training resolution, the DINOv2 backbone's ``position_embeddings``
    parameter has an incompatible shape.  ``load_state_dict(strict=False)`` does **not**
    skip shape mismatches on matching keys — it raises ``RuntimeError``.

    This function bicubic-interpolates every PE tensor in the checkpoint whose shape
    differs from the target grid, modifying *checkpoint_state* in-place before
    ``load_state_dict`` is called.

    Args:
        checkpoint_state: The ``"model"`` sub-dict from a loaded checkpoint.
        pe_size: Target grid side length in patches (number of patches per spatial
            dimension, assuming a square grid).  Typically
            ``model_config.positional_encoding_size``.
    """
    n_target = pe_size * pe_size  # target number of patch tokens

    pe_keys = [k for k in checkpoint_state if k.endswith(_PE_KEY_SUFFIX)]
    for key in pe_keys:
        ckpt_pe = checkpoint_state[key]  # [1, N_src+1, dim]
        n_source = ckpt_pe.shape[1] - 1  # exclude class token
        if n_source == n_target:
            continue  # no mismatch — skip

        h_src = int(math.isqrt(n_source))
        h_tgt = int(math.isqrt(n_target))
        if h_src * h_src != n_source or h_tgt * h_tgt != n_target:
            logger.warning(
                f"Skipping PE interpolation for {key}:"
                f" grid size is not a perfect square (source {n_source}, target {n_target}).",
            )
            continue

        dim = ckpt_pe.shape[-1]
        class_token = ckpt_pe[:, :1]  # [1, 1, dim] — keeps the sequence dimension
        patch_pe = ckpt_pe[:, 1:]  # [1, N_src, dim]

        patch_pe = patch_pe.reshape(1, h_src, h_src, dim).permute(0, 3, 1, 2)  # [1, dim, H, W]
        patch_pe = F.interpolate(
            patch_pe.float(),
            size=(h_tgt, h_tgt),
            mode="bicubic",
            align_corners=False,
            antialias=patch_pe.device.type != "mps",
        ).to(ckpt_pe.dtype)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, n_target, dim)  # [1, N_tgt, dim]

        checkpoint_state[key] = torch.cat([class_token, patch_pe], dim=1)
        logger.debug(
            "Interpolated positional embeddings %s: %s → %s.",
            key,
            tuple(ckpt_pe.shape),
            tuple(checkpoint_state[key].shape),
        )


def _extract_tensor_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a tensor-only state dict from common DINOv2 checkpoint formats.

    Args:
        checkpoint: Object returned by :func:`torch.load`.

    Returns:
        Mapping of string keys to tensors.

    Raises:
        ValueError: If no tensor state dict can be found.
    """
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "teacher", "student"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
                return {str(k): v for k, v in value.items() if torch.is_tensor(v)}
        if any(torch.is_tensor(v) for v in checkpoint.values()):
            return {str(k): v for k, v in checkpoint.items() if torch.is_tensor(v)}
    raise ValueError("DINOv2 checkpoint does not contain a tensor state dict.")


def _load_dinov2_checkpoint_file(checkpoint_path: str) -> Any:
    """Load a DINOv2 checkpoint from ``.pth``/``.pt`` or ``.safetensors``."""
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors DINOv2 checkpoints requires safetensors. "
                "Install the project dependencies or convert the checkpoint to .pth."
            ) from exc
        return load_file(checkpoint_path, device="cpu")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _strip_known_dinov2_prefix(key: str) -> str:
    """Strip common wrapper prefixes from a DINOv2 checkpoint key."""
    prefixes = (
        "module.",
        "model.",
        "teacher.",
        "student.",
        "backbone.0.encoder.",
        "backbone.0.",
        "backbone.",
        "dinov2.",
        "dinov2_with_registers.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _split_qkv(tensor: torch.Tensor, target_name: str) -> torch.Tensor:
    """Split an original DINOv2 qkv tensor into query/key/value tensors."""
    part = target_name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
    if tensor.shape[0] % 3 != 0:
        raise ValueError(f"qkv tensor first dimension is not divisible by 3: {tuple(tensor.shape)}")
    q, k, v = tensor.chunk(3, dim=0)
    if part == "query":
        return q
    if part == "key":
        return k
    if part == "value":
        return v
    raise ValueError(f"Cannot split qkv tensor for target {target_name!r}.")


def _map_original_dinov2_key(key: str) -> list[tuple[str, str | None]]:
    """Map original DINOv2 repository keys to RF-DETR backbone encoder keys.

    Args:
        key: Prefix-stripped source key.

    Returns:
        List of ``(target_key, transform)`` pairs. ``transform`` is ``"qkv"``
        for packed query/key/value tensors and ``None`` otherwise.
    """
    if key == "cls_token":
        return [("embeddings.cls_token", None)]
    if key == "mask_token":
        return [("embeddings.mask_token", None)]
    if key == "pos_embed":
        return [("embeddings.position_embeddings", None)]
    if key == "register_tokens":
        return [("embeddings.register_tokens", None)]
    if key.startswith("patch_embed.proj."):
        return [(key.replace("patch_embed.proj.", "embeddings.patch_embeddings.projection.", 1), None)]
    if key.startswith("blocks."):
        parts = key.split(".")
        if len(parts) < 4:
            return []
        layer_idx = parts[1]
        suffix = ".".join(parts[2:])
        base = f"encoder.layer.{layer_idx}"
        if suffix.startswith("attn.qkv."):
            qkv_suffix = suffix.removeprefix("attn.qkv.")
            return [
                (f"{base}.attention.attention.query.{qkv_suffix}", "qkv"),
                (f"{base}.attention.attention.key.{qkv_suffix}", "qkv"),
                (f"{base}.attention.attention.value.{qkv_suffix}", "qkv"),
            ]
        replacements = (
            ("attn.proj.", "attention.output.dense."),
            ("ls1.gamma", "layer_scale1.lambda1"),
            ("ls2.gamma", "layer_scale2.lambda1"),
        )
        for source, target in replacements:
            if suffix.startswith(source):
                return [(f"{base}.{suffix.replace(source, target, 1)}", None)]
            if suffix == source:
                return [(f"{base}.{target}", None)]
        return [(f"{base}.{suffix}", None)]
    if key.startswith("norm."):
        return [(key.replace("norm.", "layernorm.", 1), None)]
    return []


def _map_hf_dinov2_key(key: str) -> list[tuple[str, str | None]]:
    """Map Hugging Face DINOv2 keys to RF-DETR backbone encoder keys."""
    if key.startswith("embeddings.") or key.startswith("encoder.layer.") or key.startswith("layernorm."):
        return [(key, None)]
    return []


def _target_key_candidates(target_key: str, target_state: dict[str, torch.Tensor]) -> list[str]:
    """Return possible target keys for bare and wrapped DINOv2 encoders."""
    candidates = [target_key]
    wrapped = f"encoder.{target_key}"
    if wrapped not in candidates:
        candidates.append(wrapped)
    return [key for key in candidates if key in target_state]


def _target_prefix(target_state: dict[str, torch.Tensor]) -> str:
    """Return the wrapper prefix used by the target DINOv2 state dict."""
    if any(key.startswith("encoder.embeddings.") for key in target_state):
        return "encoder."
    return ""


def _find_transformer_block_count(module: torch.nn.Module) -> int:
    """Return the number of transformer blocks in wrapped/unwrapped DINOv2 modules."""
    current: Any = module
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        layer = getattr(current, "layer", None)
        if layer is not None:
            try:
                return len(layer)
            except TypeError:
                return 0
        current = getattr(current, "encoder", None)
    return 0


def _module_loaded_ratio(
    target_state: dict[str, torch.Tensor],
    loaded_keys: set[str],
    prefixes: tuple[str, ...],
) -> float:
    """Compute loaded parameter ratio for target keys starting with *prefixes*."""
    total = sum(t.numel() for k, t in target_state.items() if k.startswith(prefixes))
    loaded = sum(target_state[k].numel() for k in loaded_keys if k in target_state and k.startswith(prefixes))
    if total == 0:
        return 0.0
    return loaded / total


def _log_dinov2_backbone_config(encoder: torch.nn.Module, checkpoint_path: str) -> None:
    """Log RF-DETR DINOv2 architecture fields before loading."""
    config = getattr(encoder, "config", None)
    logger.info("DINOv2 backbone checkpoint: %s", checkpoint_path)
    if config is None:
        logger.info("RF-DETR backbone config unavailable on encoder type %s", type(encoder).__name__)
        return
    logger.info(
        "RF-DETR backbone config: hidden_size=%s depth=%s num_heads=%s mlp_ratio=%s patch_size=%s "
        "image_size=%s positional_encoding_shape=%s num_register_tokens=%s cls_token=%s",
        getattr(config, "hidden_size", None),
        getattr(config, "num_hidden_layers", None),
        getattr(config, "num_attention_heads", None),
        getattr(config, "mlp_ratio", None),
        getattr(config, "patch_size", None),
        getattr(config, "image_size", None),
        tuple(getattr(getattr(encoder, "embeddings", None), "position_embeddings").shape)
        if hasattr(getattr(encoder, "embeddings", None), "position_embeddings")
        else None,
        getattr(config, "num_register_tokens", None),
        hasattr(getattr(encoder, "embeddings", None), "cls_token"),
    )


def _checkpoint_config_value(checkpoint: Any, name: str) -> Any:
    """Return a config value from common checkpoint config containers."""
    if not isinstance(checkpoint, dict):
        return None
    for config_key in ("config", "model_config", "args"):
        config = checkpoint.get(config_key)
        if isinstance(config, dict) and name in config:
            return config[name]
        if config is not None and hasattr(config, name):
            return getattr(config, name)
    return None


def _source_dinov2_summary(checkpoint: Any, source_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Infer source DINOv2 architecture fields from config and tensor shapes."""
    stripped = {_strip_known_dinov2_prefix(key): tensor for key, tensor in source_state.items()}
    patch_weight = stripped.get("patch_embed.proj.weight")
    if patch_weight is None:
        patch_weight = stripped.get("embeddings.patch_embeddings.projection.weight")
    pos_embed = stripped.get("pos_embed")
    if pos_embed is None:
        pos_embed = stripped.get("embeddings.position_embeddings")
    cls_token = stripped.get("cls_token")
    if cls_token is None:
        cls_token = stripped.get("embeddings.cls_token")
    register_tokens = stripped.get("register_tokens")
    if register_tokens is None:
        register_tokens = stripped.get("embeddings.register_tokens")
    block_indexes: list[int] = []
    for key in stripped:
        parts = key.split(".")
        if len(parts) > 1 and key.startswith("blocks.") and parts[1].isdigit():
            block_indexes.append(int(parts[1]))
        if len(parts) > 2 and key.startswith("encoder.layer.") and parts[2].isdigit():
            block_indexes.append(int(parts[2]))
    hidden_size = _checkpoint_config_value(checkpoint, "hidden_size")
    if hidden_size is None and cls_token is not None:
        hidden_size = cls_token.shape[-1]
    return {
        "variant": _checkpoint_config_value(checkpoint, "variant")
        or _checkpoint_config_value(checkpoint, "model_name")
        or "unknown",
        "embed_dim": hidden_size,
        "depth": _checkpoint_config_value(checkpoint, "num_hidden_layers")
        or ((max(block_indexes) + 1) if block_indexes else None),
        "num_heads": _checkpoint_config_value(checkpoint, "num_attention_heads"),
        "mlp_ratio": _checkpoint_config_value(checkpoint, "mlp_ratio"),
        "patch_size": _checkpoint_config_value(checkpoint, "patch_size")
        or (patch_weight.shape[-1] if patch_weight is not None else None),
        "patch_embed.proj.weight": tuple(patch_weight.shape) if patch_weight is not None else None,
        "pos_embed": tuple(pos_embed.shape) if pos_embed is not None else None,
        "cls_token": tuple(cls_token.shape) if cls_token is not None else None,
        "register_tokens": tuple(register_tokens.shape) if register_tokens is not None else None,
    }


def _validate_source_config_against_target(source_summary: dict[str, Any], encoder: torch.nn.Module) -> None:
    """Validate source DINOv2 config fields when they are available."""
    target_config = getattr(encoder, "config", None)
    if target_config is None:
        return
    comparisons = (
        ("embed_dim", "hidden_size"),
        ("depth", "num_hidden_layers"),
        ("num_heads", "num_attention_heads"),
        ("patch_size", "patch_size"),
    )
    mismatches = []
    for source_key, target_key in comparisons:
        source_value = source_summary.get(source_key)
        target_value = getattr(target_config, target_key, None)
        if source_value is None or target_value is None:
            logger.warning("DINOv2 source %s unavailable; relying on tensor shape checks.", source_key)
            continue
        if int(source_value) != int(target_value):
            mismatches.append(f"{source_key}: source={source_value} target={target_value}")
    if mismatches:
        raise RuntimeError("DINOv2 source/RF-DETR backbone config mismatch: " + "; ".join(mismatches))


def _validate_loaded_dinov2_backbone(report: DinoV2BackboneLoadReport) -> None:
    """Raise when critical DINOv2 backbone weights were not loaded."""
    required_modules = ("patch_embed", "pos_embed", "cls_token", "transformer_blocks", "norm")
    missing_modules = [name for name in required_modules if report.module_ratios.get(name, 0.0) <= 0.0]
    missing_register_tokens = any("register_tokens" in key for key in report.missing_keys)
    if report.module_ratios.get("register_tokens", 0.0) == 0.0 and missing_register_tokens:
        missing_modules.append("register_tokens")
    if missing_modules:
        raise RuntimeError(f"Critical DINOv2 backbone modules were not loaded: {missing_modules}")
    if report.missing_keys:
        raise RuntimeError(f"Unexpected missing DINOv2 backbone keys: {report.missing_keys}")
    if report.shape_mismatched_keys:
        raise RuntimeError(f"DINOv2 backbone shape mismatches: {report.shape_mismatched_keys}")
    if report.loaded_blocks != report.total_blocks:
        raise RuntimeError(f"Loaded only {report.loaded_blocks}/{report.total_blocks} DINOv2 transformer blocks.")


def load_dinov2_backbone_weights(nn_model: torch.nn.Module, checkpoint_path: str) -> DinoV2BackboneLoadReport:
    """Load only DINOv2 encoder weights into an RF-DETR model.

    The target module is ``nn_model.backbone[0].encoder``. Decoder, query,
    projector/neck, class head, and box head parameters are never consumed.
    Both Hugging Face-style DINOv2 keys and original DINOv2 repository keys are
    supported. Original packed ``attn.qkv`` tensors are split into RF-DETR's
    ``query``, ``key``, and ``value`` tensors.

    Args:
        nn_model: RF-DETR ``LWDETR`` instance.
        checkpoint_path: DINOv2 backbone checkpoint path.

    Returns:
        Structured loading report.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
        RuntimeError: If critical backbone weights are missing or incompatible.
        ValueError: If the checkpoint format is unsupported.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"DINOv2 backbone checkpoint not found: {checkpoint_path}")

    encoder = nn_model.backbone[0].encoder
    _log_dinov2_backbone_config(encoder, checkpoint_path)
    checkpoint = _load_dinov2_checkpoint_file(checkpoint_path)
    source_state = _extract_tensor_state_dict(checkpoint)
    source_summary = _source_dinov2_summary(checkpoint, source_state)
    logger.info("DINOv2 source checkpoint summary: %s", source_summary)
    _validate_source_config_against_target(source_summary, encoder)
    target_state = encoder.state_dict()
    load_state: dict[str, torch.Tensor] = {}
    consumed_source: set[str] = set()
    shape_mismatched: dict[str, str] = {}
    skipped: dict[str, str] = {}

    for raw_key, tensor in source_state.items():
        key = _strip_known_dinov2_prefix(raw_key)
        mapped = _map_hf_dinov2_key(key) or _map_original_dinov2_key(key)
        if not mapped:
            skipped[raw_key] = "not_a_dinov2_backbone_key_or_unsupported_mapping"
            continue
        for target_key, transform in mapped:
            target_candidates = _target_key_candidates(target_key, target_state)
            if not target_candidates:
                skipped[raw_key] = f"mapped_target_absent:{target_key}"
                continue
            resolved_target_key = target_candidates[0]
            try:
                value = _split_qkv(tensor, target_key) if transform == "qkv" else tensor
            except ValueError as exc:
                shape_mismatched[resolved_target_key] = str(exc)
                continue
            if tuple(value.shape) != tuple(target_state[resolved_target_key].shape):
                shape_mismatched[resolved_target_key] = (
                    f"source {raw_key} shape {tuple(value.shape)} "
                    f"!= target {tuple(target_state[resolved_target_key].shape)}"
                )
                continue
            load_state[resolved_target_key] = value.to(dtype=target_state[resolved_target_key].dtype)
            consumed_source.add(raw_key)

    missing = sorted(k for k in target_state if k not in load_state)
    unexpected = sorted(k for k in source_state if k not in consumed_source and k not in skipped)
    incompatible = encoder.load_state_dict(load_state, strict=False)
    strict_missing = sorted(getattr(incompatible, "missing_keys", []))
    strict_unexpected = sorted(getattr(incompatible, "unexpected_keys", []))
    if strict_unexpected:
        unexpected.extend(strict_unexpected)
    if strict_missing != missing:
        logger.warning("DINOv2 loader internal missing-key mismatch: computed=%s strict=%s", missing, strict_missing)

    loaded_keys = sorted(load_state)
    loaded_key_set = set(loaded_keys)
    prefix = _target_prefix(target_state)
    total_blocks = _find_transformer_block_count(encoder)
    loaded_blocks = len(
        {
            int(key.removeprefix(prefix).split(".")[2])
            for key in loaded_keys
            if key.removeprefix(prefix).startswith("encoder.layer.")
            and len(key.removeprefix(prefix).split(".")) > 2
            and key.removeprefix(prefix).split(".")[2].isdigit()
        }
    )
    module_ratios = {
        "patch_embed": _module_loaded_ratio(target_state, loaded_key_set, (f"{prefix}embeddings.patch_embeddings.",)),
        "pos_embed": _module_loaded_ratio(target_state, loaded_key_set, (f"{prefix}embeddings.position_embeddings",)),
        "cls_token": _module_loaded_ratio(target_state, loaded_key_set, (f"{prefix}embeddings.cls_token",)),
        "register_tokens": _module_loaded_ratio(target_state, loaded_key_set, (f"{prefix}embeddings.register_tokens",)),
        "transformer_blocks": _module_loaded_ratio(target_state, loaded_key_set, (f"{prefix}encoder.layer.",)),
        "norm": _module_loaded_ratio(target_state, loaded_key_set, (f"{prefix}layernorm.",)),
    }
    backbone_params = sum(p.numel() for p in nn_model.backbone.parameters())
    loaded_backbone_params = sum(target_state[k].numel() for k in loaded_keys)
    report = DinoV2BackboneLoadReport(
        checkpoint_path=checkpoint_path,
        source_key_count=len(source_state),
        loaded_keys=loaded_keys,
        missing_keys=missing,
        unexpected_keys=sorted(set(unexpected)),
        shape_mismatched_keys=shape_mismatched,
        skipped_keys=skipped,
        total_model_params=sum(p.numel() for p in nn_model.parameters()),
        trainable_params=sum(p.numel() for p in nn_model.parameters() if p.requires_grad),
        backbone_params=backbone_params,
        loaded_backbone_params=loaded_backbone_params,
        module_ratios=module_ratios,
        loaded_blocks=loaded_blocks,
        total_blocks=total_blocks,
    )
    for line in report.to_log_lines():
        logger.info(line)
    _validate_loaded_dinov2_backbone(report)
    return report


@deprecated(
    target=True,
    args_mapping={"train_config": None},
    deprecated_in="1.8",
    remove_in="1.9",
    num_warns=-1,
    stream=functools.partial(warnings.warn, category=DeprecationWarning),
)
def load_pretrain_weights(
    nn_model: torch.nn.Module,
    model_config: ModelConfig,
    train_config: TrainConfig | None = None,
) -> List[str]:
    """Load pretrained checkpoint weights into *nn_model* in-place.

    Canonical implementation shared by the L1 facade (``_build_model_context``
    in ``rfdetr.detr``) and the L2 LightningModule (``RFDETRModelModule.__init__``
    in ``rfdetr.training.module_model``).

    Uses the Pydantic-aware logic from ``module_model.py``:

    - When the user did **not** explicitly override ``num_classes`` (left at the
      ModelConfig default), the checkpoint class count is treated as authoritative
      and the model head is auto-aligned to it.
    - When the user **did** explicitly override ``num_classes`` to a value larger
      than the checkpoint provides, the head is temporarily aligned to the
      checkpoint for loading, then expanded back to the configured size.
    - When the checkpoint has more classes than configured (backbone-pretrain
      scenario), both reinitializations are applied: expand to checkpoint size for
      loading, then trim to configured size.

    Class names stored in the checkpoint ``args`` are extracted and returned.

    Args:
        nn_model: The model whose weights will be updated in-place.
        model_config: Pydantic ``ModelConfig`` instance. Must have
            ``pretrain_weights``, ``num_classes``, ``num_queries``, and
            ``group_detr`` attributes.
        train_config: Deprecated since v1.8 — no longer used internally.
            Passing a non-``None`` value emits a ``DeprecationWarning``.
            Omit the argument; it will be removed in v1.9.

    Returns:
        List of class name strings from the checkpoint, or an empty list if none
        are present or if ``model_config.pretrain_weights`` is ``None``.

    Raises:
        Exception: If the checkpoint file cannot be loaded even after a re-download.
    """
    mc = model_config
    pretrain_weights = mc.pretrain_weights
    if pretrain_weights is None:
        return []
    class_names: List[str] = []

    # Download first (no-op if already present and hash is valid).
    download_pretrain_weights(pretrain_weights)
    # If the first download attempt didn't produce the file (e.g. stale MD5
    # caused an earlier ValueError that was silently swallowed), retry with
    # MD5 validation disabled so a stale registry hash can't block training.
    if not os.path.isfile(pretrain_weights):
        logger.warning("Pretrain weights not found after initial download; retrying without MD5 validation.")
        download_pretrain_weights(pretrain_weights, redownload=True, validate_md5=False)
    validate_pretrain_weights(pretrain_weights, strict=False)

    try:
        checkpoint = torch.load(pretrain_weights, map_location="cpu", weights_only=False)
    except Exception:
        logger.info("Failed to load pretrain weights, re-downloading")
        download_pretrain_weights(pretrain_weights, redownload=True, validate_md5=False)
        checkpoint = torch.load(pretrain_weights, map_location="cpu", weights_only=False)

    # Normalize PyTorch Lightning native .ckpt format to the expected {"model": {...}}
    # structure.  PTL stores model weights in "state_dict" with keys prefixed by
    # "model." (matching the attribute path inside RFDETRModelModule).  Legacy and
    # BestModelCallback checkpoints already have a top-level "model" key.
    if "model" not in checkpoint and "state_dict" in checkpoint:
        logger.debug("Normalizing PTL .ckpt checkpoint format (state_dict -> model)")
        prefix = "model."
        # When the model was wrapped with torch.compile, PTL stores weights with keys
        # like "model._orig_mod.<param>".  Strip the extra "_orig_mod." segment so the
        # resulting keys match the expected bare parameter names.
        compile_prefix = "_orig_mod."
        model_state = {}
        for k, v in checkpoint["state_dict"].items():
            if k.startswith(prefix):
                stripped = k[len(prefix) :]
                if stripped.startswith(compile_prefix):
                    stripped = stripped[len(compile_prefix) :]
                model_state[stripped] = v
        if not model_state:
            raise ValueError(
                f"The checkpoint at {pretrain_weights!r} appears to be in PyTorch Lightning "
                "format ('state_dict' key present, 'model' key absent), but 'state_dict' "
                "contains no keys with the expected 'model.' prefix. "
                "The checkpoint may be corrupt or in an unsupported format."
            )
        checkpoint["model"] = model_state
        # PTL stores training hyper-parameters under "hyper_parameters".  Map them
        # to the "args" key expected by class-name extraction and compatibility checks
        # (only when "args" is not already present).
        if "args" not in checkpoint and "hyper_parameters" in checkpoint:
            checkpoint["args"] = checkpoint["hyper_parameters"]

    # Extract class_names from the checkpoint if available (ported from detr.py).
    if "args" in checkpoint:
        raw_class_names = _ckpt_args_get(checkpoint["args"], "class_names")
        if raw_class_names:
            # Normalize to a new List[str] to avoid leaking mutable references and
            # to respect the annotated return type.
            if isinstance(raw_class_names, str):
                class_names = [raw_class_names]
            else:
                try:
                    iterator = iter(raw_class_names)
                except TypeError:
                    # Non-iterable, ignore and keep the default empty list.
                    class_names = []
                else:
                    class_names = [name for name in iterator if isinstance(name, str)]

    validate_checkpoint_compatibility(checkpoint, mc)

    # Determine whether the user explicitly set num_classes on the ModelConfig,
    # and whether that explicit value differs from the model default.
    user_set_num_classes = False
    if hasattr(mc, "model_fields_set"):
        user_set_num_classes = "num_classes" in getattr(mc, "model_fields_set", set())
    default_num_classes = type(mc).model_fields["num_classes"].default
    num_classes = mc.num_classes
    # True only when the user explicitly set num_classes to a non-default value.
    user_overrode_default_num_classes = user_set_num_classes and num_classes != default_num_classes

    checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    configured_num_classes_plus_bg = num_classes + 1
    if checkpoint_num_classes != configured_num_classes_plus_bg:
        # Align model head size before loading checkpoint weights.
        if checkpoint_num_classes < configured_num_classes_plus_bg:
            # Checkpoint has FEWER classes than configured.
            if not user_overrode_default_num_classes:
                # Auto-align to the checkpoint when the user did NOT provide a
                # non-default override for num_classes (i.e., left it at the
                # ModelConfig default): treat the checkpoint as authoritative.
                num_classes = checkpoint_num_classes - 1
                configured_num_classes_plus_bg = checkpoint_num_classes
                mc.num_classes = num_classes
        # In all mismatch cases we need the head to match the checkpoint's
        # class count so load_state_dict succeeds without size mismatches.
        nn_model.reinitialize_detection_head(checkpoint_num_classes)

    # Reshape query embeddings to the configured query count, preserving per-group
    # structure when the checkpoint records its training-time num_queries / group_detr.
    # See _slice_query_param_per_group for why a flat slice is wrong with group_detr > 1.
    ckpt_args = checkpoint.get("args")
    ckpt_num_queries_raw = _ckpt_args_get(ckpt_args, "num_queries") if ckpt_args is not None else None
    ckpt_group_detr_raw = _ckpt_args_get(ckpt_args, "group_detr") if ckpt_args is not None else None
    try:
        ckpt_num_queries = int(ckpt_num_queries_raw) if ckpt_num_queries_raw is not None else None
        ckpt_group_detr = int(ckpt_group_detr_raw) if ckpt_group_detr_raw is not None else None
    except (TypeError, ValueError):
        logger.warning(
            "load_pretrain_weights: checkpoint args.num_queries / args.group_detr not coercible "
            "to int; falling back to legacy flat slice."
        )
        ckpt_num_queries = None
        ckpt_group_detr = None
    # When exactly one of the pair is present, infer the missing value from the
    # first matching tensor's shape.  This handles PTL checkpoints where
    # BestModelCallback writes TrainConfig.model_dump() into checkpoint["args"]
    # but TrainConfig does not include num_queries (it lives on ModelConfig).
    if (ckpt_num_queries is None) != (ckpt_group_detr is None):
        _first_query_key = next(
            (k for k in checkpoint["model"] if any(k.endswith(s) for s in _QUERY_PARAM_SUFFIXES)),
            None,
        )
        if _first_query_key is not None:
            _n = checkpoint["model"][_first_query_key].shape[0]
            _absent: str | None = None
            if ckpt_num_queries is not None and ckpt_num_queries > 0 and _n % ckpt_num_queries == 0:
                ckpt_group_detr = _n // ckpt_num_queries
                _absent, _inferred, _known, _known_val = "group_detr", ckpt_group_detr, "num_queries", ckpt_num_queries
            elif ckpt_group_detr is not None and ckpt_group_detr > 0 and _n % ckpt_group_detr == 0:
                ckpt_num_queries = _n // ckpt_group_detr
                _absent, _inferred, _known, _known_val = "num_queries", ckpt_num_queries, "group_detr", ckpt_group_detr
            if _absent is not None:
                logger.warning(
                    "load_pretrain_weights: args.%s absent; inferred ckpt_%s=%d from tensor rows %d ÷ ckpt_%s=%d.",
                    _absent,
                    _absent,
                    _inferred,
                    _n,
                    _known,
                    _known_val,
                )
    # Warn once (not once per suffix key) when falling back to the legacy flat slice.
    if mc.group_detr > 1 and (ckpt_num_queries is None or ckpt_group_detr is None):
        logger.warning(
            "load_pretrain_weights: checkpoint lacks args.num_queries / "
            "args.group_detr; falling back to flat slice. With "
            "group_detr=%d this may scramble per-group query structure if "
            "the checkpoint was trained with group_detr > 1.",
            mc.group_detr,
        )
    for name in list(checkpoint["model"].keys()):
        if any(name.endswith(x) for x in _QUERY_PARAM_SUFFIXES):
            tensor = checkpoint["model"][name]
            if ckpt_num_queries is not None and ckpt_group_detr is not None:
                checkpoint["model"][name] = _slice_query_param_per_group(
                    tensor,
                    ckpt_num_queries=ckpt_num_queries,
                    ckpt_group_detr=ckpt_group_detr,
                    target_num_queries=mc.num_queries,
                    target_group_detr=mc.group_detr,
                )
            else:
                # Legacy checkpoint with no num_queries/group_detr in args:
                # preserve the original flat slice for backward compatibility.
                # NOTE: the flat slice is incorrect for group_detr > 1 — it scrambles
                # groups 1+ when num_queries decreases. Legacy checkpoints predate
                # multi-group training, so in practice they are all group_detr == 1.
                checkpoint["model"][name] = tensor[: mc.num_queries * mc.group_detr]

    interpolate_position_embeddings(checkpoint["model"], mc.positional_encoding_size)
    incompatible = nn_model.load_state_dict(checkpoint["model"], strict=False)
    _warn_on_partial_load(incompatible, pretrain_weights)

    # If the user explicitly set a class count larger than the checkpoint,
    # expand/reinitialize the head back to the configured size after load.
    if checkpoint_num_classes < configured_num_classes_plus_bg and user_overrode_default_num_classes:
        nn_model.reinitialize_detection_head(configured_num_classes_plus_bg)

    # Only trim back down when loading a larger pretrain checkpoint into a
    # smaller configured task-specific class count.
    if num_classes + 1 < checkpoint_num_classes:
        nn_model.reinitialize_detection_head(num_classes + 1)

    return class_names


def apply_lora(nn_model: torch.nn.Module) -> None:
    """Apply LoRA adapters to the backbone encoder of *nn_model*.

    Replaces ``nn_model.backbone[0].encoder`` in-place with a PEFT-wrapped
    encoder using DoRA with rank 16 and alpha 16.

    Args:
        nn_model: LWDETR model whose backbone encoder will receive LoRA adapters.

    Raises:
        ImportError: If ``peft`` is not installed.
            Install via the RF-DETR extras, for example::

                pip install "rfdetr[lora]"
                # or
                pip install "rfdetr[train]"
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requires the 'peft' dependency. "
            "Install it via RF-DETR extras, e.g.: "
            'pip install "rfdetr[lora]" or pip install "rfdetr[train]".'
        ) from exc

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        use_dora=True,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "qkv",
            "query",
            "key",
            "value",
            "cls_token",
            "register_tokens",
        ],
    )
    nn_model.backbone[0].encoder = get_peft_model(nn_model.backbone[0].encoder, lora_config)
