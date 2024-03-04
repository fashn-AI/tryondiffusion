import torch
from torchvision.transforms import ToPILImage


def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255


def tensor_to_pil(img: torch.Tensor, unnormalize: bool = False) -> torch.Tensor:
    """
    Process a single image tensor by optionally unnormalizing and converting it to a PIL image.

    :param img: A tensor representing an image.
    :param unnormalize: A flag to apply unnormalization, default is False.
    :return: A PIL image.
    """
    if unnormalize:
        img = unnormalize_zero_to_one(img)
    to_pil = ToPILImage()
    return to_pil(img)


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5
