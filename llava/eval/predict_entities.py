import argparse
import torch
import os
import json
from datetime import datetime
from PIL import Image
from copy import deepcopy
import warnings
from tqdm import tqdm
import re
from collections import defaultdict
import pathlib

from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images
from llava.eval.model_vqa import preprocess_qwen

from llava.preprocessing.extract_image_crops import create_crops
from llava.eval.metrics import Scorer, extract_result
from llava.preprocessing.convert_annotations_to_llava import convert_to_llava_format
from llava.riv_cot_utils import load_model_in_eval, rescale_bbox, format_normalized_to_coco_bbox, save_json

warnings.filterwarnings("ignore")


def extract_entities(text):
    """
    Extract entity names and bounding box coordinates from model output text.
    Expected format: 'entity_name [x1, y1, x2, y2]'

    Args:
        text (str): Text containing entities with bounding box coordinates.

    Returns:
        list: A list of dictionaries with entity names as keys and bounding box coordinates as values.
    """
    # Define the regex pattern to capture entity names and bounding box coordinates
    pattern = r'([a-zA-Z0-9\s/]+)\s*\[(\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+)\]'

    # Find all matches in the input text
    matches = re.findall(pattern, text)

    # Convert matches to a dictionary, parsing coordinates as lists of floats
    entities = []

    for name, bbox in matches:
        # Normalize the entity name by stripping whitespace
        name = name.strip()

        # Check if bbox is in correct format by trying to parse to floats
        try:
            coords = list(map(float, bbox.split(',')))
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                if coords != [0.0, 0.0, 0.0, 0.0] and x1 <= x2 and y1 <= y2:
                    entities.append({name: coords})
                else:
                    print(f"Discarding invalid bbox for entity '{name}': {coords}")
        except ValueError:
            # Skip this entry if parsing fails or incorrect number of coordinates
            print(f"Failing to parse '{name}:{bbox}")
            continue

    return entities


def predict_entities(model_path, llava_format_path, dataset, image_folder, test_json_path, conv_mode, temperature,
                     max_new_tokens, device, crop_method, extend_by_percent):
    """
    Runs the model to predict entity bounding boxes and processes them.

    Parameters:
        model_path (str): Path to the LLaVA model.
        llava_format_path (str): Path to the dataset in LLaVA format.
        dataset (str): Dataset name.
        image_folder (str): Path to the image folder.
        test_json_path (str): Path to the test JSON file.
        conv_mode (str): Conversation mode.
        temperature (float): Temperature for text generation.
        max_new_tokens (int): Maximum number of new tokens to generate.
        device (str): Compute device.
        crop_method (str): Cropping method for bounding boxes.
        extend_by_percent (int or None): Percentage by which to extend bounding boxes.
    """
    # Load model
    tokenizer, model, image_processor = load_model_in_eval(model_path)

    # Load LLaVA format file
    with open(os.path.expanduser(llava_format_path), 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Load test data
    with open(test_json_path, 'r') as f:
        bbox_test_data = json.load(f)

    for line in tqdm(data):
        # Build prompt
        qs = line["conversations"][0]["value"]
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = [line["conversations"][0], {'from': 'gpt', 'value': None}]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        input_ids = preprocess_qwen(prompt, tokenizer, has_image=True).cuda()

        # Process images
        line["image"] = [line["image"][0]]  # Keep only the full image
        try:
            images = [Image.open(os.path.join(image_folder, image_file)) for image_file in line["image"]]
            image_tensors = process_images(images, image_processor, model.config)
        except TypeError as e:
            print(f"Skipping image due to error: {e}")  # Log the error and continue
            continue  # Skip this image and proceed with the next one
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
        image_sizes = [image.size for image in images]

        # Generate prediction
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        # Extract entities from model output
        predicted_entities = extract_entities(outputs)

        # Rescale bounding box coordinates
        rescaled_predicted_entities = []
        width, height = image_sizes[0]
        rescaled_predicted_entities = [{name: format_normalized_to_coco_bbox(bbox, width, height)}
                                       for entity in predicted_entities for name, bbox in entity.items()]

        # Update test data with predicted entities
        bbox_test_data[line['id']]['original_relevant_entities'] = rescaled_predicted_entities

    # Save bounding box predictions
    output_dir = os.path.join(model_path, f'eval_{crop_method}_{extend_by_percent}' if extend_by_percent is not None else f'eval_{crop_method}')
    bbox_json_path = os.path.join(output_dir, 'pred_bboxes.json')
    save_json(bbox_test_data, bbox_json_path)
    print(f"Saving bboxes predictions to {bbox_json_path}")

    # generate crops
    print("Generating crops")
    create_crops(crop_method, extend_by_percent, bbox_json_path, output_dir, image_folder)

    # Convert to LLaVA format
    print('Converting to LLaVA format')
    convert_to_llava_format(
        input_file=bbox_json_path,
        dataset=dataset,
        image_folder=pathlib.Path(output_dir).joinpath(f'crops_{crop_method}_{extend_by_percent}' if extend_by_percent else f'crops_{crop_method}'),
        prompt_format='QP-RB-RV-EA',
        explanation_type='original',
        output_dir=output_dir,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="drivingvqa")
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--llava_format_path", type=str, default="")
    parser.add_argument("--test_json_path", type=str, default="")
    parser.add_argument("--conv_mode", type=str, default="qwen_1_5")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--crop_method", type=str, default='normal', choices=['normal', 'normal_30', 'normal_50', 'normal_70', 'visual_cot'])
    args = parser.parse_args()

    # Parse crop method and optional extend percent
    match = re.match(r"([a-zA-Z]+(?:_[a-zA-Z]+)?)(?:_(\d+))?", args.crop_method)
    crop_method = match.group(1)
    extend_percent = int(match.group(2)) if match.group(2) else None

    predict_entities(
        model_path=args.model_path,
        llava_format_path=args.llava_format_path,
        dataset=args.dataset,
        image_folder=args.image_folder,
        test_json_path=args.test_json_path,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        crop_method=crop_method,
        extend_by_percent=extend_percent
    )

if __name__ == "__main__":
    main()
