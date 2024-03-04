import torch
from torch.utils.data import DataLoader, Dataset

from tryondiffusion import TryOnImagen, TryOnImagenTrainer, get_unet_by_name

TRAIN_UNET_NUMBER = 1
BASE_UNET_IMAGE_SIZE = (128, 128)
SR_UNET_IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
NUM_ITERATIONS = 2
TIMESTEPS = (2, 2)


class SyntheticTryonDataset(Dataset):
    def __init__(self, num_samples, image_size, pose_size=(18, 2)):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            image_size (tuple): The height and width of the images (height, width).
            pose_size (tuple): The size of the pose tensors (default: (18, 2)).
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.pose_size = pose_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        person_image = torch.randn(3, *self.image_size)
        ca_image = torch.randn(3, *self.image_size)
        garment_image = torch.randn(3, *self.image_size)
        person_pose = torch.randn(*self.pose_size)
        garment_pose = torch.randn(*self.pose_size)

        sample = {
            "person_images": person_image,
            "ca_images": ca_image,
            "garment_images": garment_image,
            "person_poses": person_pose,
            "garment_poses": garment_pose,
        }

        return sample


def tryondiffusion_collate_fn(batch):
    return {
        "person_images": torch.stack([item["person_images"] for item in batch]),
        "ca_images": torch.stack([item["ca_images"] for item in batch]),
        "garment_images": torch.stack([item["garment_images"] for item in batch]),
        "person_poses": torch.stack([item["person_poses"] for item in batch]),
        "garment_poses": torch.stack([item["garment_poses"] for item in batch]),
    }


def main():
    print("Instantiating the dataset and dataloader...")
    dataset = SyntheticTryonDataset(
        num_samples=1000, image_size=SR_UNET_IMAGE_SIZE if TRAIN_UNET_NUMBER == 2 else BASE_UNET_IMAGE_SIZE
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=tryondiffusion_collate_fn,
    )
    validation_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=tryondiffusion_collate_fn,
    )
    print("Checking the dataset and dataloader...")
    sample = next(iter(train_dataloader))
    for k, v in sample.items():
        print(f"{k}: {v.shape}")

    # Instantiate the unets
    print("Instantiating U-Nets...")
    base_unet = get_unet_by_name("base")
    sr_unet = get_unet_by_name("sr")

    # Instantiate the Imagen model
    imagen = TryOnImagen(
        unets=(base_unet, sr_unet),
        image_sizes=(BASE_UNET_IMAGE_SIZE, SR_UNET_IMAGE_SIZE),
        timesteps=TIMESTEPS,
    )

    print("Instantiating the trainer...")
    trainer = TryOnImagenTrainer(
        imagen=imagen,
        max_grad_norm=1.0,
        accelerate_cpu=True,
        accelerate_gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    trainer.add_train_dataloader(train_dataloader)
    trainer.add_valid_dataloader(validation_dataloader)

    print("Starting training loop...")
    # training loop
    for i in range(NUM_ITERATIONS):
        # TRAINING
        loss = trainer.train_step(unet_number=TRAIN_UNET_NUMBER)
        print(f"loss: {loss}")
        valid_loss = trainer.valid_step(unet_number=TRAIN_UNET_NUMBER)
        print(f"valid loss: {valid_loss}")

    # SAMPLING
    print("Starting sampling loop...")
    validation_sample = next(trainer.valid_dl_iter)
    _ = validation_sample.pop("person_images")
    imagen_sample_kwargs = dict(
        **validation_sample,
        batch_size=BATCH_SIZE,
        cond_scale=2.0,
        start_at_unet_number=1,
        return_all_unet_outputs=True,
        return_pil_images=True,
        use_tqdm=True,
        use_one_unet_in_gpu=True,
    )
    images = trainer.sample(**imagen_sample_kwargs)  # returns List[Image]
    assert len(images) == 2
    assert len(images[0]) == BATCH_SIZE and len(images[1]) == BATCH_SIZE

    for unet_output in images:
        for image in unet_output:
            image.show()


if __name__ == "__main__":
    # python ./examples/test_tryon_imagen_trainer.py
    main()
