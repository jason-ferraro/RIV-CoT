import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from typing import List

from llava.mm_utils import process_images
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model

def xywh_to_xyxy(box):
    """Convert COCO format [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]."""
    x_min, y_min, width, height = box
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]


def normalize_bbox(sub_image_info, image_width, image_height):
    """
    Normalize bounding box coordinates to the range [0, 1].

    Parameters:
    - sub_image_info: tuple or list (x_min, y_min, x_max, y_max).
    - image_width: int, width of the image.
    - image_height: int, height of the image.

    Returns:
    - A tuple of normalized bounding box coordinates (x_min_norm, y_min_norm, x_max_norm, y_max_norm).
    """
    x_min, y_min, x_max, y_max = sub_image_info
    x_min_norm = x_min / image_width
    y_min_norm = y_min / image_height
    x_max_norm = x_max / image_width
    y_max_norm = y_max / image_height
    return (x_min_norm, y_min_norm, x_max_norm, y_max_norm)


def rescale_bbox(norm_bbox, image_width, image_height):
    """
    Denormalize bounding box coordinates from the range [0, 1] to original image dimensions.

    Parameters:
    - norm_bbox: tuple or list (x_min_norm, y_min_norm, x_max_norm, y_max_norm).
    - image_width: int, width of the image.
    - image_height: int, height of the image.

    Returns:
    - A tuple of denormalized bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = norm_bbox
    x_min = int(x_min_norm * image_width)
    y_min = int(y_min_norm * image_height)
    x_max = int(x_max_norm * image_width)
    y_max = int(y_max_norm * image_height)
    return (x_min, y_min, x_max, y_max)


def format_normalized_to_coco_bbox(normalized_bbox, img_width, img_height):
    """
    Convert a normalized bounding box [xmin, ymin, xmax, ymax] to COCO format [x, y, width, height].

    Parameters:
    - normalized_bbox (list): Normalized bounding box [xmin, ymin, xmax, ymax] between 0 and 1.
    - img_width (int): Width of the image.
    - img_height (int): Height of the image.

    Returns:
    - list: COCO bounding box [x, y, width, height].
    """
    xmin, ymin, xmax, ymax = normalized_bbox
    x = xmin * img_width
    y = ymin * img_height
    width = (xmax - xmin) * img_width
    height = (ymax - ymin) * img_height
    return [x, y, width, height]


def format_coco_to_normalized_bbox(bbox: List[float], img_width: int, img_height: int) -> str:
    """
    Convert a COCO bounding box to normalized [xmin, ymin, xmax, ymax] format.

    Parameters:
    - coco_bbox (list): Bounding box in COCO format [x, y, width, height].
    - img_width (int): Width of the image.
    - img_height (int): Height of the image.

    Returns:
    - list: Normalized bounding box [xmin, ymin, xmax, ymax] between 0 and 1.
    """
    x, y, width, height = bbox
    xmin = x / img_width
    ymin = y / img_height
    xmax = (x + width) / img_width
    ymax = (y + height) / img_height
    return [xmin, ymin, xmax, ymax]


def cropwithbbox(pil_img, sub_image_info):
    """
    Adapt the croped image to a square suited to CLIP ViT
    Args:
        pil_img:
        sub_image_info:

    Returns:
        cropped_region
    """
    width, height = pil_img.size
    x_min, y_min, x_max, y_max = sub_image_info
    if sum([x_min, y_min, x_max, y_max]) < 5:
        x_min = x_min * max(width, height)
        y_min = y_min * max(width, height)
        x_max = x_max * max(width, height)
        y_max = y_max * max(width, height)
    if width > height:
        overlay = (width - height) // 2
        y_min = max(0, y_min - overlay)
        y_max = max(0, y_max - overlay)
    else:
        overlay = (height - width) // 2
        x_min = max(0, x_min - overlay)
        x_max = max(0, x_max - overlay)
    center_point = [(x_min + x_max)//2, (y_min + y_max)//2]
    half_sizes = [(x_max - x_min)//2, (y_max - y_min)//2]
    cropped_half_size = max(max(half_sizes), 112)
    upper_left_point = [center_point[0]-cropped_half_size, center_point[1]-cropped_half_size]
    if upper_left_point[0] < 0:
        center_point[0] += (-upper_left_point[0])
    if upper_left_point[1] < 0:
        center_point[1] += (-upper_left_point[1])
    lower_right_point = [center_point[0]+cropped_half_size, center_point[1]+cropped_half_size]
    if lower_right_point[0] > width:
        center_point[0] -= (lower_right_point[0] - width)
    if lower_right_point[1] > height:
        center_point[1] -= (lower_right_point[1] - height)
    cropped_region = [max(0, center_point[0]-cropped_half_size), max(0, center_point[1]-cropped_half_size), min(width, center_point[0]+cropped_half_size), min(height, center_point[1]+cropped_half_size)]
    cropped_image = pil_img.crop(cropped_region)
    return cropped_image

def reformat_data_dict(data):
    new_data = defaultdict()
    for i in range(len(data)):
        new_data[i] = data[i][list(data[i].keys())[0]]
    return new_data


def extract_sub_image(image, bbox, extend_by_percent=0):
    """
    Extracts a sub-image from the given PIL image based on a bounding box in COCO format,
    with an option to extend the bounding box by a specified percentage.

    Parameters:
    - image: PIL Image object
    - bbox: list of bounding box coordinates in COCO format [x, y, width, height]
    - extend_by_percent: float, percentage to extend the bounding box on each side

    Returns:
    - sub_image: PIL Image object of the cropped region
    """
    if extend_by_percent is None:
        extend_by_percent = 0  # No extension if None is provided

    x, y, width, height = map(int, bbox)  # Ensure coordinates are integers
    img_width, img_height = image.size

    # Calculate the extension in pixels
    extend_x = int(width * extend_by_percent / 100)
    extend_y = int(height * extend_by_percent / 100)

    # Adjust bounding box with the extension, making sure it stays within image bounds
    x_min = max(0, x - extend_x)
    y_min = max(0, y - extend_y)
    x_max = min(img_width, x + width + extend_x)
    y_max = min(img_height, y + height + extend_y)

    # Crop the image using the adjusted bounding box
    sub_image = image.crop((x_min, y_min, x_max, y_max))

    return sub_image

def save_json(data, file_path):
    """
    Saves dictionary or list data to a JSON file.

    Parameters:
        data (dict or list): Data to save.
        file_path (str): Path to the JSON file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved: {file_path}")

def load_model_in_eval(model_path):
    """
    Loads the pre-trained LLaVA model and tokenizer.

    Parameters:
        model_path (str): Path to the model.

    Returns:
        tuple: (tokenizer, model, image_processor)
    """
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_size = next((size for size in ["0.5b", "7b", "70b"] if size in model_path.lower()), None)
    overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064} if model_size == "7b" else None

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen", overwrite_config=overwrite_config
    )
    model.eval()
    return tokenizer, model, image_processor
