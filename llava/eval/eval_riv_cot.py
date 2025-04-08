import argparse
import torch
import os
import json
import re
from datetime import datetime
from PIL import Image
from copy import deepcopy
import warnings
from tqdm import tqdm

from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images
from llava.eval.model_vqa import preprocess_qwen

from llava.eval.metrics import Scorer, extract_result
from llava.riv_cot_utils import load_model_in_eval, save_json, format_normalized_to_coco_bbox, extract_sub_image, cropwithbbox

warnings.filterwarnings("ignore")


def extract_last_bbox(text):
    """
    Extract the last bounding box coordinates from text.
    Expected format: 'text [x1, y1, x2, y2]'

    Args:
        text (str): Text to analyze

    Returns:
        list or None: Last bounding box coordinates as a list of floats if found, None otherwise
    """
    # Define the regex pattern to capture bounding box coordinates
    pattern = r'\[([-]?\d+\.?\d*,\s*[-]?\d+\.?\d*,\s*[-]?\d+\.?\d*,\s*[-]?\d+\.?\d*)\]'

    # Find all matches in the input text
    matches = re.finditer(pattern, text)
    last_match = None
    bbox_pattern_found = False

    # Get the last match
    for match in matches:
        last_match = match
        bbox_pattern_found = True

    if last_match:
        # Extract the bbox coordinates
        bbox_str = last_match.group(1)
        try:
            # Convert string coordinates to float list
            coords = [float(coord.strip()) for coord in bbox_str.split(',')]

            # Verify we have exactly 4 coordinates
            if len(coords) == 4:
                # Basic validation of coordinates
                if (coords != [0.0, 0.0, 0.0, 0.0] and
                    all(0 <= x <= 1 for x in coords) and
                    coords[0] <= coords[2] and
                    coords[1] <= coords[3]):
                    return coords, bbox_pattern_found
        except ValueError:
            pass

    return None, bbox_pattern_found

def eval_multistep(model_path, llava_format_path, image_folder, conv_mode, temperature, max_new_tokens, crop_method,
                   extend_by_percent, device):
    """Evaluate the model using a multi-step approach for entity extraction.

    Parameters:
        model_path (str): Path to the LLaVA model.
        llava_format_path (str): Path to the dataset in LLaVA format.
        image_folder (str): Path to the image folder.
        conv_mode (str): Conversation mode.
        temperature (float): Temperature for text generation.
        max_new_tokens (int): Maximum number of new tokens to generate.
        no_image (bool): Whether to disable image input.
        device (str): Device to use ('cuda' or 'cpu').
    """
    scorer = Scorer()
    raw_preds, all_preds, all_true_answers = [], {}, {}

    # Load model
    tokenizer, model, image_processor = load_model_in_eval(model_path)

    # Load LLaVA format file
    with open(os.path.expanduser(llava_format_path), 'r', encoding='utf-8') as f:
            data = json.load(f)

    for line in tqdm(data):
        # Build prompt
        qs = line["conversations"][0]["value"]
        conv = deepcopy(conv_templates[conv_mode])
        conv.append_message(conv.roles[1], None)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        prompt = [line["conversations"][0]]
        try:
            images = [Image.open(os.path.join(image_folder, line["image"][0]))]
            image_tensors = process_images(images, image_processor, model.config)
        except TypeError as e:
            print(f"Skipping image due to error: {e}")
            continue

        img_crop, predicted_bbox, bbox_pattern_found = True, False, False
        steps = 0

        while img_crop:
            steps += 1
            if not bbox_pattern_found:
                prompt.append({'from': 'gpt', 'value': None})
            # Build image
            if isinstance(img_crop, Image.Image):
                images.append(img_crop)
            input_ids = preprocess_qwen(prompt, tokenizer, has_image=True).cuda()
            image_tensors = process_images(images, image_processor, model.config)
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
            predicted_bbox, bbox_pattern_found = extract_last_bbox(outputs)
            #print(f'===>>> Step {steps}')
            #print(outputs)

            # Rescale bounding box coordinates
            if predicted_bbox:
                # Convert normalized coordinates to COCO format
                width, height = image_sizes[0][0], image_sizes[0][1]
                coco_bbox = format_normalized_to_coco_bbox(predicted_bbox, width, height)
                rescaled_bbox = [round(coord, 2) for coord in coco_bbox]
                #print('Predicted bbox:', rescaled_bbox)

                if crop_method == 'normal':
                    img_crop = extract_sub_image(images[0], rescaled_bbox, extend_by_percent)
                elif crop_method == 'visual_cot':
                    img_crop = cropwithbbox(images[0], normalized_bbox)

                # Update prompt with cropped image
                prompt[-1]['value'] = prompt[-1]['value'] + outputs if prompt[-1]['value'] else outputs
                prompt.append({'from': 'human', 'value':' <image>'})
                bbox_pattern_found = False
            elif bbox_pattern_found:
                # Continue generation if bbox pattern was found but invalid
                prompt[-1]['value'] = prompt[-1]['value'] + outputs if prompt[-1]['value'] else outputs
                img_crop = True
            else:
                prompt[-1]['value'] = prompt[-1]['value'] + outputs if prompt[-1]['value'] else outputs
                img_crop = False
                #print('\n')

        # Save raw predictions
        raw_preds.append(
            {
                "id": line["id"],
                "conversations": [{"from": message["from"], "value": message["value"]} for message in prompt]
            }
        )

        # Extract results
        all_preds[line["id"]] = extract_result(outputs)
        all_true_answers[line["id"]] = extract_result(line["conversations"][-1]["value"])

    # Save predictions
    output_dir = os.path.join(model_path, f'eval_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    save_json(raw_preds, os.path.join(output_dir, 'outputs.json'))
    save_json(all_preds, os.path.join(output_dir, 'predictions.json'))
    save_json(all_true_answers, os.path.join(output_dir, 'true_answers.json'))

    # Compute and save scores
    scores = scorer.compute_score(os.path.join(output_dir, 'predictions.json'), os.path.join(output_dir, 'true_answers.json'))
    save_json(scores, os.path.join(output_dir, 'scores.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--llava_format_path", type=str)
    parser.add_argument("--conv_mode", type=str, default="qwen_1_5")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--crop_method", type=str, default='normal', choices=['normal', 'normal_30', 'normal_50', 'normal_70', 'visual_cot'])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Parse crop method and optional extend percent
    match = re.match(r"([a-zA-Z]+(?:_[a-zA-Z]+)?)(?:_(\d+))?", args.crop_method)
    crop_method = match.group(1)
    extend_percent = int(match.group(2)) if match.group(2) else None

    eval_multistep(
        model_path=args.model_path,
        llava_format_path=args.llava_format_path,
        image_folder=args.image_folder,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        crop_method=crop_method,
        extend_by_percent=extend_percent,
        device=args.device
    )

if __name__ == "__main__":
    main()
