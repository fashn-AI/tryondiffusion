import torch

from tryondiffusion import get_unet_by_name

IMAGE_SIZE = (128, 128)  # (height, width), change to (256, 256) for "sr" unet)
BATCH_SIZE = 2
MODEL_NAME = "base"  # "base" or "sr"
MODEL_KWARGS = {}
LOWRES_IMAGE = MODEL_NAME == "sr"


def main():
    unet = get_unet_by_name(MODEL_NAME, **MODEL_KWARGS)
    print(f"Model: {unet.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in unet.parameters()) / 1e9:.2f}B")

    noisy_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE)
    lowres_cond_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE) if LOWRES_IMAGE else None

    ca_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE)
    garment_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE)
    person_poses = torch.randn(
        BATCH_SIZE,
        18,
        2,
    )
    garment_poses = torch.randn(
        BATCH_SIZE,
        18,
        2,
    )

    time = torch.randn(
        BATCH_SIZE,
    )
    lowres_noise_times = torch.randn(BATCH_SIZE) if LOWRES_IMAGE else None

    ca_noise_times = torch.randn(
        BATCH_SIZE,
    )
    garment_noise_times = torch.randn(
        BATCH_SIZE,
    )

    print(f"noisy_images shape: {noisy_images.shape}")
    output = unet(
        noisy_images=noisy_images,
        time=time,
        lowres_cond_img=lowres_cond_images,
        lowres_noise_times=lowres_noise_times,
        ca_images=ca_images,
        ca_noise_times=ca_noise_times,
        garment_images=garment_images,
        garment_noise_times=garment_noise_times,
        person_poses=person_poses,
        garment_poses=garment_poses,
    )
    print(f"output shape: {output.shape}")


if __name__ == "__main__":
    # python ./examples/test_tryon_imagen.py
    main()
