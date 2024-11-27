import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import os
from PIL import Image
from models.definitions.vgg_nets import Vgg16, Vgg19, Vgg16Experimental

IMAGENET_MEAN_255=[0.485, 0.456, 0.406]
IMAGENET_STD_NEUTRAL=[0.229, 0.224, 0.225]


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def prepare_img_from_pil(image: Image.Image, device: torch.device):
    """
    Prepares an image from a PIL.Image object for use in PyTorch models.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    original_width, original_height = image.size
    new_width = 250
    new_height = int((new_width / original_width) * original_height)
    target_shape = (new_height, new_width)

    transform = transforms.Compose([
        transforms.Resize(target_shape),  # Resize to the given shape
        transforms.ToTensor(),
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


def prepare_model(content_feature_map_index, model, device):
    # we are not tuning model weights -> we are only tuning optimizing_img's pixels! (that's why requires_grad=False)
    experimental = False
    if model == 'vgg16':
        if experimental:
            # much more flexible for experimenting with different style representations
            model = Vgg16Experimental(content_feature_map_index, requires_grad=False, show_progress=True)
        else:
            model = Vgg16(content_feature_map_index, requires_grad=False, show_progress=True)
    elif model == 'vgg19':
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


def to_image_format(tensor):
    """
    Converts a PyTorch tensor or NumPy array to a uint8 image format suitable for display.
    Assumes the input tensor/array has values in a non-standard range (e.g., -90 to 90).
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()  # Move to CPU and convert to NumPy array

    # Ensure dimensions are correct (e.g., C x H x W to H x W x C)
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # Channels-first
        tensor = np.transpose(tensor, (1, 2, 0)) 

    # Normalize to range [0, 255]
    tensor_min, tensor_max = tensor.min(), tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * 255.0
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
