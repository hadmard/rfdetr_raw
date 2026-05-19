from rfdetr import RFDETRBase


def main() -> None:
    dataset_dir = "/home/zju/Desktop/cv/rf-detr/data/strawberry_uv_coco"
    dinov2_backbone_ckpt = "/home/zju/Desktop/cv/rf-detr/weights/dinov2-small/model.safetensors"

    # No RF-DETR pretrain weights: backbone is partially seeded from DINOv2,
    # but projector/transformer/decoder/heads are randomly initialised.
    # Align the positional-encoding grid with the actual input resolution
    # (560 / patch_size 14 = 40) so the PE doesn't have to be bicubic-resampled
    # from the 37x37 grid every forward pass.
    model = RFDETRBase(
        pretrain_weights=None,
        dinov2_backbone_ckpt=dinov2_backbone_ckpt,
        positional_encoding_size=40,
    )

    # Defaults in TrainConfig (lr_vit_layer_decay=0.8, lr_component_decay=0.7,
    # warmup_epochs=0) are tuned for finetuning a fully pretrained RF-DETR ckpt.
    # For from-scratch training we have to disable the layer/component decay,
    # raise the LR, and add a real warmup, otherwise the backbone trains at ~1e-6
    # while the random heads run at full LR and the loss collapses.
    model.train(
        dataset_dir=dataset_dir,
        output_dir="output/strawberry_uv_backbone_only_200e_b8_v2",
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
