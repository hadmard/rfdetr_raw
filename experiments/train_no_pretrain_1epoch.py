# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Train RF-DETR without RF-DETR or DINOv2 pretrained weights."""

from rfdetr import RFDETRBase


def main() -> None:
    """Train RF-DETR with random initialization."""
    model = RFDETRBase(
        pretrain_weights=None,
        positional_encoding_size=40,
    )

    model.train(
        dataset_dir="data/strawberry_uv_coco",
        output_dir="output/strawberry_uv_no_pretrain_200e_b8_g6_lr5e4_flat",
        epochs=200,
        batch_size=8,
        grad_accum_steps=6,
        num_workers=12,
        lr=5e-4,
        lr_encoder=5e-4,
        lr_vit_layer_decay=1.0,
        lr_component_decay=1.0,
        lr_scheduler="cosine",
        lr_min_factor=0.05,
        warmup_epochs=5.0,
        devices=2,
        strategy="ddp_find_unused_parameters_true",
        progress_bar="tqdm",
        tensorboard=False,
        run_test=False,
    )


if __name__ == "__main__":
    main()
