import torch
from torchvision import transforms
from PIL import Image

def save_normalized_tensor_as_rgb_image(tensor, save_path):
    """
    Save a normalized tensor as an RGB image with pixel values in the range [0, 255].

    Args:
        tensor (torch.Tensor): The normalized tensor with shape [3, 224, 224].
        save_path (str): The file path to save the RGB image.

    Returns:
        None
    """
    # Denormalize the tensor to range [0, 1]
    denormalized_tensor = tensor.clone()
    denormalized_tensor[0] = denormalized_tensor[0] * 0.229 + 0.485
    denormalized_tensor[1] = denormalized_tensor[1] * 0.224 + 0.456
    denormalized_tensor[2] = denormalized_tensor[2] * 0.225 + 0.406

    # Convert the tensor to the range [0, 255]
    denormalized_tensor = (denormalized_tensor * 255).clamp(0, 255).to(torch.uint8)

    # Convert the tensor to PIL image
    pil_image = transforms.ToPILImage()(denormalized_tensor)

    # Save the PIL image to the file path
    pil_image.save(save_path)

import os

def create_directory_if_not_exists(directory_path):
    # Check if the directory path exists
    if not os.path.exists(directory_path):
        # If the directory does not exist, create it along with any necessary parent directories
        os.makedirs(directory_path)
    return directory_path