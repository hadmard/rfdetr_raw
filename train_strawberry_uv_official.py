# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Base on the strawberry UV dataset from scratch."""

from rfdetr import RFDETRBase


DATASET_DIR = "/home/zju/Desktop/cv/rf-detr/data/strawberry_uv_coco"
OUTPUT_DIR = "output/strawberry_uv_no_pretrain_200e_b8_g6_lr5e-4_fixed_accum_flat"
PEAK_LR = 5e-4


def main() -> None:
    """Run official RFDETR.train() with flat learning rates for scratch training."""
    print(f"dataset_dir={DATASET_DIR}")
    print(f"output_dir={OUTPUT_DIR}")
    print(f"scratch peak lr={PEAK_LR:g} for encoder/decoder/head")

    model = RFDETRBase(
        pretrain_weights=None,
        dinov2_backbone_ckpt=None,
        num_classes=3,
        resolution=560,
        positional_encoding_size=40,
    )

    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=200,
        batch_size=8,
        grad_accum_steps=6,
        num_workers=12,
        lr=PEAK_LR,
        lr_encoder=PEAK_LR,
        lr_vit_layer_decay=1.0,
        lr_component_decay=1.0,
        lr_scheduler="cosine",
        lr_min_factor=0.1,
        warmup_epochs=5.0,
        devices=2,
        strategy="ddp_find_unused_parameters_true",
        progress_bar="tqdm",
        tensorboard=False,
        run_test=False,
    )


if __name__ == "__main__":
    main()
