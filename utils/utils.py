import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from models.definitions.vgg_nets import Vgg16, Vgg19, Vgg16_GradCAM, Vgg19_GradCAM


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def prepare_img_from_pil(image: Image.Image, device: torch.device, img_width):
    """
    Prepares an image from a PIL.Image object for use in PyTorch models.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    original_width, original_height = image.size
    new_width = img_width
    new_height = int((new_width / original_width) * original_height)
    target_shape = (new_height, new_width)

    transform = transforms.Compose([
        transforms.Resize(target_shape),  # Resize to the given shape
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),  # Scale to [0, 255]
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    
    # Transform the image
    img_tensor = transform(image).to(device).unsqueeze(0)
    
    return img_tensor


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')


def prepare_model(content_feature_map_index, model, device, gradCAM=False):
    # we are not tuning model weights -> we are only tuning optimizing_img's pixels! (that's why requires_grad=False)
    if model == 'vgg16':
        if gradCAM:
            model = Vgg16_GradCAM(content_feature_map_index, requires_grad=False, show_progress=True)
        else:
            model = Vgg16(content_feature_map_index, requires_grad=False, show_progress=True)
    elif model == 'vgg19':
        if gradCAM:
            model = Vgg19_GradCAM(content_feature_map_index, requires_grad=False, show_progress=True)
        else:
            model = Vgg19(content_feature_map_index, requires_grad=False, show_progress=True)
    else:
        raise ValueError(f'{model} not supported.')

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def to_image_format(tensor, gamma=1.3):
    """
    Converts a PyTorch tensor or NumPy array to a uint8 image format suitable for display.
    Includes gamma correction to reduce brightness and improve contrast.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()  # Move to CPU and convert to NumPy array

    # Ensure dimensions are correct (e.g., C x H x W to H x W x C)
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # Channels-first
        tensor = np.transpose(tensor, (1, 2, 0))

    # Normalize to range [0, 1]
    tensor_min, tensor_max = tensor.min(), tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    # Apply gamma correction to control brightness (gamma > 1 for less brightness, < 1 for more)
    tensor = np.power(tensor, gamma)

    # Scale to [0, 255]
    tensor = tensor * 255.0
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=255, neginf=0)
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    return tensor


def grad_cam(activations, gradients, image_size):
    # Calculate the weights for each channel (average of the gradients across spatial dimensions)
    weights = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)

    # Create the Grad-CAM map by multiplying activations with the weights
    grad_cam_map = torch.sum(weights * activations, dim=1, keepdim=True)

    # Apply ReLU to keep only positive influences
    grad_cam_map = torch.relu(grad_cam_map)

    # Upsample the Grad-CAM map to the size of the original image
    grad_cam_map = torch.nn.functional.interpolate(grad_cam_map, size=image_size, mode='bilinear', align_corners=False)

    return grad_cam_map