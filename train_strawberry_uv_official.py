from rfdetr import RFDETRBase


def main() -> None:
    dataset_dir = "/home/zju/Desktop/cv/rf-detr/data/strawberry_uv_coco"
    dinov2_backbone_ckpt = "/home/zju/Desktop/cv/rf-detr/weights/dinov2-small/model.safetensors"

    model = RFDETRBase(
        pretrain_weights=None,
        dinov2_backbone_ckpt=dinov2_backbone_ckpt,
    )

    model.train(
        dataset_dir=dataset_dir,
        output_dir="output/strawberry_uv_backbone_only_200e_b8",
        epochs=1,
        batch_size=8,
        grad_accum_steps=1,
        num_workers=0,
        lr=1e-4,
        lr_encoder=1.5e-4,
        lr_scheduler="cosine",
        lr_min_factor=0.01,
        warmup_epochs=0.0,
        devices=2,
        strategy="ddp_find_unused_parameters_true",
        progress_bar="tqdm",
        tensorboard=False,
        run_test=False,
    )


if __name__ == "__main__":
    main()
