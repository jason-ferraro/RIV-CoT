import json
import torch
import argparse
import os
import re
import warnings
from tqdm import tqdm
from PIL import Image
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection


def run_grounding_dino_on_image(sample, image, model, processor, device):
    """
    Runs GroundingDINO on a given image to detect relevant entities.

    Args:
        sample (dict): Dictionary containing sample information including inferred entities.
        image (PIL.Image): The image to process.
        model (torch.nn.Module): Pretrained GroundingDINO model.
        processor (transformers.Processor): Processor for text and image inputs.
        device: PyTorch device object.

    Returns:
        list: A list of detected relevant entities and their bounding boxes.
    """
    inferred_entity_labels = sample.get('inferred_entities', [])

    if not inferred_entity_labels:
        print(f"No entities to detect in sample with img {sample['img_filename']}.")
        return []

    entity_queries = " . ".join(inferred_entity_labels) + " ."

    inputs = processor(images=image, text=entity_queries, return_tensors="pt").to(model.device)

    with torch.no_grad():
        try:
            outputs = model(**inputs)
        except Exception as error:
            warnings.warn(f"Error processing {sample.get('id', 'Unknown')}: {error}")
            return []

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]

    # Keep only the top 5 detected entities (if necessary)
    filtered_entity_predictions = sorted(
        zip(results['boxes'], results['labels'], results['scores']),
        key=lambda x: x[2], reverse=True
    )[:5]

    # Extract bounding boxes  and save them in COCO format
    detected_entities = []
    for bbox, label, score in filtered_entity_predictions:
        # Convert to COCO format [x_min, y_min, width, height]
        x_min, y_min, x_max, y_max = bbox.tolist()
        formatted_bbox = [round(x_min, 2), round(y_min, 2), round(x_max - x_min, 2), round(y_max - y_min, 2)]

        # Format entity names
        formatted_label = re.sub(r'\s*-\s*', '-', label).strip()

        detected_entities.append({formatted_label: formatted_bbox})

    return detected_entities


def process_dataset(input_path, output_path, image_folder, model, text_image_processor, device):
    """
    Processes an entire dataset by running GroundingDINO on images to extract relevant entity bounding boxes.

    Args:
        input_path (str): Path to the input dataset JSON file.
        output_path (str): Path to save the output annotations.
        image_folder (str): Path to the folder containing image files.
        model (torch.nn.Module): Loaded GroundingDINO model.
        text_image_processor (transformers.Processor): Loaded processor for the model.
        device: PyTorch device object.
    """
    # Load the dataset
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    print("Starting entity detection using GroundingDINO...")

    # Process each sample in the dataset
    for sample_id, sample in tqdm(data.items()):
        image_path = os.path.join(image_folder, sample['img_filename'])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            warnings.warn(f"Could not open image {image_path}: {error}")
            continue

        # Get annotations
        sample['relevant_entities'] = run_grounding_dino_on_image(sample, image, model, text_image_processor, device)

    # Save the annotations in COCO format
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)

    print(f"Detected entities annotations have been saved to: {output_path}")

# Main function to handle argument parsing
def main():
    parser = argparse.ArgumentParser(description="Entity Detection and Bounding Box Extraction using GroundingDINO")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON dataset")
    parser.add_argument("--output", type=str, required=True, help="Path for the output JSON file with bounding boxes")
    parser.add_argument("--image_folder", type=str, required=True, help="Directory where the images are stored")
    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-base", help="HuggingFace model ID for GroundingDINO")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cuda or cpu)")
    args = parser.parse_args()

    # Initialize the grounding DINO model and processor
    print("Loading GroundingDINO model and processor...")
    text_image_processor = GroundingDinoProcessor.from_pretrained(args.model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model_id).to(args.device)

    # Process the dataset
    process_dataset(args.input, args.output, args.image_folder, model, text_image_processor, args.device)

if __name__ == "__main__":
    main()
