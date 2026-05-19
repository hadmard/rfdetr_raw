# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Run controlled RF-DETR initialization experiments with diagnostics.

This script keeps official RF-DETR model construction and ``RFDETR.train()``
as the formal training path. It adds preflight diagnostics, dry-run, and a
small overfit probe before handing off to the official train interface.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from rfdetr.config import TrainConfig
from rfdetr.training import RFDETRDataModule, RFDETRModelModule
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _variant_class(name: str) -> type:
    """Return an RF-DETR variant class by short name."""
    from rfdetr.variants import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    variants = {
        "base": RFDETRBase,
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    if name not in variants:
        raise ValueError(f"Unknown variant {name!r}; choose from {sorted(variants)}")
    return variants[name]


def _run_text(command: list[str]) -> str:
    """Run a command and return stripped stdout or an error marker."""
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True)
    except OSError as exc:
        return f"<failed: {exc}>"
    if result.returncode != 0:
        return f"<exit {result.returncode}: {result.stderr.strip()}>"
    return result.stdout.strip()


def _count_params(params: list[torch.nn.Parameter]) -> int:
    """Count elements in a list of parameters."""
    return sum(p.numel() for p in params)


def _print_environment_report() -> None:
    """Print reproducibility information for the current run."""
    logger.info("RF-DETR commit hash: %s", _run_text(["git", "rev-parse", "HEAD"]))
    logger.info("RF-DETR git diff:\n%s", _run_text(["git", "diff", "--", "."]))
    logger.info("Python: %s", platform.python_version())
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("CUDA version: %s", torch.version.cuda)
    if torch.cuda.is_available():
        logger.info("GPU name: %s", torch.cuda.get_device_name(0))


def _print_dataset_report(datamodule: RFDETRDataModule, dataset_dir: str) -> None:
    """Print dataset size and class mapping diagnostics."""
    datamodule.setup("fit")
    train_ds = datamodule._dataset_train
    val_ds = datamodule._dataset_val
    logger.info("dataset path: %s", dataset_dir)
    logger.info("train image count: %s", len(train_ds) if train_ds is not None else None)
    logger.info("val image count: %s", len(val_ds) if val_ds is not None else None)
    logger.info("class names: %s", datamodule.class_names)
    coco = getattr(train_ds, "coco", None)
    cat2label = getattr(train_ds, "cat2label", None)
    label2cat = getattr(train_ds, "label2cat", None)
    logger.info("cat2label: %s", cat2label)
    logger.info("label2cat: %s", label2cat)
    has_background = "__background__" in (datamodule.class_names or [])
    logger.info("contains background class in dataset categories: %s", has_background)
    if coco is not None:
        logger.info("raw COCO categories: %s", {k: v.get("name") for k, v in sorted(coco.cats.items())})


def _print_config_report(model_module: RFDETRModelModule, train_config: TrainConfig) -> None:
    """Print model/training hyperparameter diagnostics."""
    mc = model_module.model_config
    logger.info("num_classes: %s", mc.num_classes)
    logger.info("image resolution: %s", mc.resolution)
    logger.info("batch size: %s", train_config.batch_size)
    logger.info("epochs: %s", train_config.epochs)
    logger.info("optimizer: AdamW")
    logger.info("scheduler: %s", train_config.lr_scheduler)
    logger.info("warmup epochs: %s", train_config.warmup_epochs)
    logger.info("backbone learning rate lr_encoder: %s", train_config.lr_encoder)
    logger.info("decoder/head learning rate lr: %s", train_config.lr)
    logger.info("weight decay: %s", train_config.weight_decay)
    logger.info("gradient clipping: %s", train_config.clip_max_norm)
    logger.info("AMP enabled: %s", mc.amp)
    logger.info("seed: %s", train_config.seed)
    logger.info("device: %s", mc.device)
    if mc.freeze_encoder:
        logger.warning(
            "WARNING: backbone is frozen. Since decoder/head are randomly initialized, "
            "this may cause very poor training."
        )


def _print_param_group_report(model_module: RFDETRModelModule, train_config: TrainConfig) -> None:
    """Print approximate param-group diagnostics without instantiating the PTL trainer."""
    model = model_module.model
    groups = {
        "backbone_encoder": [p for n, p in model.named_parameters() if n.startswith("backbone.0.encoder")],
        "projector_neck": [p for n, p in model.named_parameters() if n.startswith("backbone.0.projector")],
        "decoder": [p for n, p in model.named_parameters() if n.startswith("transformer.decoder")],
        "query": [p for n, p in model.named_parameters() if "query_feat" in n or "refpoint_embed" in n],
        "class_head": [p for n, p in model.named_parameters() if n.startswith("class_embed")],
        "box_head": [p for n, p in model.named_parameters() if n.startswith("bbox_embed")],
    }
    for name, params in groups.items():
        trainable = [p for p in params if p.requires_grad]
        lr = train_config.lr_encoder if name == "backbone_encoder" else train_config.lr
        logger.info(
            "param_group name=%s lr=%s weight_decay=%s params=%s trainable_params=%s frozen=%s",
            name,
            lr,
            train_config.weight_decay,
            _count_params(params),
            _count_params(trainable),
            len(trainable) == 0,
        )


def _grad_norm(module: torch.nn.Module) -> float:
    """Return global gradient norm for a module."""
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().norm().item() ** 2)
    return total**0.5


def run_dry_run(model_module: RFDETRModelModule, datamodule: RFDETRDataModule, device: torch.device) -> None:
    """Run one real-batch forward/backward diagnostic."""
    model_module.to(device)
    model_module.train()
    batch = next(iter(datamodule.train_dataloader()))
    batch = datamodule.transfer_batch_to_device(batch, device, 0)
    samples, targets = batch
    logger.info("dry_run image tensor shape: %s", tuple(samples.tensors.shape))
    features, poss = model_module.model.backbone(samples)
    logger.info("dry_run backbone output shapes: %s", [tuple(feature.tensors.shape) for feature in features])
    logger.info("dry_run positional output shapes: %s", [tuple(pos.shape) for pos in poss])
    outputs = model_module.model(samples, targets)
    logger.info("dry_run pred_logits shape: %s", tuple(outputs["pred_logits"].shape))
    logger.info("dry_run pred_boxes shape: %s", tuple(outputs["pred_boxes"].shape))
    loss_dict = model_module.criterion(outputs, targets)
    loss = sum(
        loss_dict[key] * model_module.criterion.weight_dict[key]
        for key in loss_dict
        if key in model_module.criterion.weight_dict
    )
    logger.info("dry_run loss: %s", float(loss.detach().cpu()))
    if not torch.isfinite(loss):
        raise RuntimeError("dry-run loss is NaN or Inf")
    loss.backward()
    logger.info("dry_run grad_norm backbone: %.6f", _grad_norm(model_module.model.backbone[0].encoder))
    logger.info("dry_run grad_norm decoder: %.6f", _grad_norm(model_module.model.transformer.decoder))
    logger.info("dry_run grad_norm class_head: %.6f", _grad_norm(model_module.model.class_embed))
    logger.info("dry_run grad_norm box_head: %.6f", _grad_norm(model_module.model.bbox_embed))
    if _grad_norm(model_module.model.backbone[0].encoder) == 0.0:
        raise RuntimeError("dry-run found no backbone gradient")
    model_module.zero_grad(set_to_none=True)


def run_overfit_probe(
    model_module: RFDETRModelModule,
    datamodule: RFDETRDataModule,
    device: torch.device,
    iterations: int,
) -> None:
    """Run a small repeated-batch overfit probe and require loss decrease."""
    if iterations <= 0:
        return
    model_module.to(device)
    model_module.train()
    train_ds = datamodule._dataset_train
    if train_ds is None:
        raise RuntimeError("datamodule.setup('fit') must run before overfit probe.")
    sample_count = min(16, len(train_ds))
    if sample_count < 8:
        logger.warning("overfit probe has only %d training images available; expected 8-16.", sample_count)
    subset = Subset(train_ds, list(range(sample_count)))
    loader = DataLoader(
        subset,
        batch_size=min(model_module.train_config.batch_size, sample_count),
        shuffle=True,
        collate_fn=datamodule._collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))
    batch = datamodule.transfer_batch_to_device(batch, device, 0)
    optimizer = torch.optim.AdamW(
        [p for p in model_module.parameters() if p.requires_grad],
        lr=model_module.train_config.lr,
        weight_decay=model_module.train_config.weight_decay,
    )
    losses: list[float] = []
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        samples, targets = batch
        outputs = model_module.model(samples, targets)
        loss_dict = model_module.criterion(outputs, targets)
        loss = sum(
            loss_dict[k] * model_module.criterion.weight_dict[k]
            for k in loss_dict
            if k in model_module.criterion.weight_dict
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"overfit loss became non-finite at step {step}: {loss}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % 20 == 0 or step == iterations - 1:
            logger.info("overfit_probe step=%d loss=%.6f", step, losses[-1])
    report = {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "iterations": iterations,
        "decreased": losses[-1] < losses[0] * 0.9,
    }
    logger.info("overfit_test_report: %s", json.dumps(report, indent=2))
    if not report["decreased"]:
        raise RuntimeError("overfit probe loss did not decrease by at least 10%; refusing to start full training.")


def _constructor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build model constructor kwargs for experiment A/B/C."""
    kwargs: dict[str, Any] = {"device": args.device}
    if args.experiment == "A":
        return kwargs
    if args.experiment == "B":
        kwargs["pretrain_weights"] = None
        return kwargs
    if args.experiment == "C":
        if args.dinov2_backbone_ckpt is None:
            raise ValueError("Experiment C requires --dinov2-backbone-ckpt")
        kwargs["pretrain_weights"] = None
        kwargs["dinov2_backbone_ckpt"] = args.dinov2_backbone_ckpt
        return kwargs
    raise ValueError(f"Unknown experiment {args.experiment!r}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["A", "B", "C"], required=True)
    parser.add_argument("--variant", choices=["base", "nano", "small", "medium", "large"], default="small")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-root", default="outputs/dinov2_backbone_only")
    parser.add_argument("--dinov2-backbone-ckpt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-iters", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--strategy", default="ddp_find_unused_parameters_true")
    return parser.parse_args()


def main() -> None:
    """Run diagnostics and start the selected experiment."""
    args = parse_args()
    output_dir = Path(args.output_root) / f"experiment_{args.experiment}_{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_cls = _variant_class(args.variant)
    model = variant_cls(**_constructor_kwargs(args))
    train_kwargs = {
        "dataset_dir": args.dataset_dir,
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "devices": args.devices,
        "strategy": args.strategy,
        "tensorboard": True,
        "multi_scale": True,
        "expanded_scales": True,
    }
    config = model.get_train_config(**train_kwargs)
    model._align_num_classes_from_dataset(args.dataset_dir)
    module = RFDETRModelModule(model.model_config, config)
    datamodule = RFDETRDataModule(model.model_config, config)
    _print_environment_report()
    _print_dataset_report(datamodule, args.dataset_dir)
    _print_config_report(module, config)
    _print_param_group_report(module, config)
    device = torch.device(args.device)
    run_dry_run(module, datamodule, device)
    run_overfit_probe(module, datamodule, device, args.overfit_iters)
    notes = {
        "experiment": args.experiment,
        "variant": args.variant,
        "dinov2_backbone_ckpt": args.dinov2_backbone_ckpt,
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
    }
    model.train(**train_kwargs, device=args.device, notes=notes)


if __name__ == "__main__":
    main()
