import argparse
import torch
import os
import json
from datetime import datetime
from PIL import Image
from copy import deepcopy
import warnings
from tqdm import tqdm

from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images
from llava.eval.model_vqa import preprocess_qwen

from llava.eval.metrics import Scorer, extract_result
from llava.riv_cot_utils import load_model_in_eval, save_json

warnings.filterwarnings("ignore")

def eval_model(model_path, llava_format_path, image_folder, conv_mode, temperature, max_new_tokens, no_image, device):
    """
    Evaluates the model for formats needing only a single step of generation.

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

    # Check if two-step
    two_step = 'qp-rb-rv-ea' in llava_format_path
    if two_step:
        print('Two-step conversation detected')

    for line in tqdm(data):
        # Build prompt
        qs = line["conversations"][0]["value"]
        conv = deepcopy(conv_templates[conv_mode])
        conv.append_message(conv.roles[0], qs)

        prompt = [line["conversations"][0]]
        if two_step:
            conv.append_message(conv.roles[1], line["conversations"][1]["value"])
            conv.append_message(conv.roles[0], line["conversations"][2]["value"])
            prompt += [line["conversations"][1], line["conversations"][2], {'from': 'gpt', 'value': None}]
        else:
            conv.append_message(conv.roles[1], None)
            prompt.append({'from': 'gpt', 'value': None})

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        input_ids = preprocess_qwen(prompt, tokenizer, has_image=not no_image).cuda()

        # Process images
        image_tensors, image_sizes = None, None
        if not no_image:
            if 'v' not in os.path.basename(llava_format_path): #not considering visual patch in the format
                line["image"] = [line["image"][0]]
            images = [Image.open(os.path.join(image_folder, image_file)) for image_file in line["image"]]
            try:
                image_tensors = process_images(images, image_processor, model.config)
            except TypeError as e:
                print(f"Skipping image due to error: {e}")  # Log the error and continue
                continue  # Skip this image and proceed with the next one
            image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
            image_sizes = [image.size for image in images]


        # Generate prediction
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.to(device),
                images=image_tensors if image_tensors else None,
                image_sizes=image_sizes if image_sizes else None,
                do_sample=(temperature > 0),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        # Save raw predictions
        raw_preds.append({"id": line["id"], "conversations": prompt[:-1] + [{"from": "gpt", "value": outputs}]})

        # Extract results
        all_preds[line["id"]] = extract_result(outputs)
        if two_step:
            all_true_answers[line["id"]] = extract_result(line["conversations"][3]["value"])
        else:
            all_true_answers[line["id"]] = extract_result(line["conversations"][1]["value"])

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
    parser.add_argument("--no_image", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.no_image:
        print('==/!\== Warning: Run evaluation without images')

    eval_model(
        model_path=args.model_path,
        llava_format_path=args.llava_format_path,
        image_folder=args.image_folder,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        no_image=args.no_image,
        device=args.device
    )

if __name__ == "__main__":
    main()
