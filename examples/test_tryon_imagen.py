import torch

from tryondiffusion import TryOnImagen, get_unet_by_name

IMAGE_SIZE_BASE = (128, 128)
IMAGE_SIZE_SR = (256, 256)
BATCH_SIZE = 2
TRAIN_UNET_NUMBER = 2
TIMESTEPS = (2, 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # instantiate the U-Nets
    print("Instantiating U-Nets...")
    unet1 = get_unet_by_name("base")
    unet2 = get_unet_by_name("sr")

    # prepare input images
    person_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE_SR)
    ca_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE_SR)
    garment_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE_SR)

    # prepare input poses
    person_pose = torch.randn(
        BATCH_SIZE,
        18,
        2,
    )
    garment_pose = torch.randn(
        BATCH_SIZE,
        18,
        2,
    )

    # move to device
    print(f"Moving data to {DEVICE}...")
    person_images = person_images.to(DEVICE)
    ca_images = ca_images.to(DEVICE)
    garment_images = garment_images.to(DEVICE)
    person_pose = person_pose.to(DEVICE)
    garment_pose = garment_pose.to(DEVICE)

    # instantiate the Imagen model

    imagen = TryOnImagen(
        unets=(unet1, unet2),
        image_sizes=(IMAGE_SIZE_BASE, IMAGE_SIZE_SR),
        timesteps=TIMESTEPS,
    )
    imagen = imagen.to(DEVICE)

    # forward pass
    print("Performing forward pass...")
    loss = imagen(
        person_images=person_images,
        ca_images=ca_images,
        garment_images=garment_images,
        person_poses=person_pose,
        garment_poses=garment_pose,
        unet_number=TRAIN_UNET_NUMBER,
    )

    print(f"loss: {loss}")
    print("Attempting to backpropagate...")
    loss.backward()
    print("Backpropagation successful!")

    print("Starting sampling loop...")
    images = imagen.sample(
        ca_images=ca_images,
        garment_images=garment_images,
        person_poses=person_pose,
        garment_poses=garment_pose,
        batch_size=BATCH_SIZE,
        cond_scale=2.0,
        start_at_unet_number=1,
        return_all_unet_outputs=False,
        return_pil_images=True,
        use_tqdm=True,
        use_one_unet_in_gpu=True,
    )
    images[0].show()  # or save images[0].save("output.png")


if __name__ == "__main__":
    # python ./examples/test_tryon_imagen.py
    main()
