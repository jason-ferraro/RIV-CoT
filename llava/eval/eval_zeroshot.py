import argparse
import torch
import os
import json
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor

from llava.eval.metrics import Scorer, extract_result
from llava.riv_cot_utils import save_json


def load_model_and_processor(model_name, device):
    """
    Loads a pre-trained VLM and its associated processor.

    Args:
        model_name (str): Name of the model to load.
        device (str): Device to load the model onto ("cuda" or "cpu").

    Returns:
        model (torch.nn.Module): Loaded model for generation.
        processor (transformers.Processor): Corresponding processor for input preparation.
    """
    if "Qwen" in model_name:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            f"Qwen/{model_name}",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(f"Qwen/{model_name}")
    elif "Llama" in model_name:
        from transformers import MllamaForConditionalGeneration
        model = MllamaForConditionalGeneration.from_pretrained(
            f"meta-llama/{model_name}",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(f"meta-llama/{model_name}")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model.to(device), processor


def eval_zeroshot(model_name, llava_format_path, image_folder, temperature, max_new_tokens, device):
    """
    Runs zero-shot evaluation over a dataset using a vision-language model.

    Args:
        model_name (str): Model name identifier (e.g., "Qwen2-VL-72B-Instruct").
        llava_format_path (str): Path to the LLaVA-formatted input JSON.
        image_folder (str): Directory containing images.
        temperature (float): Sampling temperature for generation.
        max_new_tokens (int): Maximum number of tokens to generate.
        device (str): Device to run model on ("cuda" or "cpu").
    """
    scorer = Scorer()
    raw_preds, all_preds, all_true_answers = [], {}, {}

    # Load model
    model, processor = load_model_and_processor(model_name, device)

    # Load LLaVA format file
    with open(os.path.expanduser(llava_format_path), 'r', encoding='utf-8') as f:
            data = json.load(f)

    for line in tqdm(data):

        if 'Qwen' in model_name:
            from qwen_vl_utils import process_vision_info
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.join(image_folder, line["image"][0])},
                    {"type": "text", "text": line["conversations"][0]["value"][8:]},
                ],
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)


        elif 'Llama' in model_name:
            image = Image.open(os.path.join(image_folder, line["image"][0]))
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": line["conversations"][0]["value"][8:]}]
            }]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(device)

        # Generate prediction
        generated_ids = model.generate(
            **inputs,
            do_sample=(temperature > 0),
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        # Decode output
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Save raw conversations
        raw_preds.append({
            "id": line["id"],
            "conversations": [
                {"from": "human", "value": line["conversations"][0]},
                {"from": "gpt", "value": outputs}
            ]
        })

        # Extract results
        all_preds[line["id"]] = extract_result(outputs)
        all_true_answers[line["id"]] = extract_result(line["conversations"][1]["value"])

    # Save predictions
    output_dir = os.path.join(model_name, f'eval_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    save_json(raw_preds, os.path.join(output_dir, 'outputs.json'))
    save_json(all_preds, os.path.join(output_dir, 'predictions.json'))
    save_json(all_true_answers, os.path.join(output_dir, 'true_answers.json'))

    # Compute and save scores
    scores = scorer.compute_score(os.path.join(output_dir, 'predictions.json'), os.path.join(output_dir, 'true_answers.json'))
    save_json(scores, os.path.join(output_dir, 'scores.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2-VL-72B-Instruct")
    parser.add_argument("--llava_format_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    eval_zeroshot(
        model_name=args.model,
        llava_format_path=args.llava_format_path,
        image_folder=args.image_folder,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )

if __name__ == "__main__":
    main()
