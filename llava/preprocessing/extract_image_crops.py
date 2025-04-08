#!/usr/bin/env python3
"""
Extract image crops based on entity bounding boxes from annotation data.

This script provides a command-line interface to the create_crops function,
allowing extraction of image crops for entities in driving theory datasets.
"""

import argparse
import os
import json
import pathlib
import shutil
from tqdm import tqdm
from PIL import Image
from llava.riv_cot_utils import xywh_to_xyxy, normalize_bbox, cropwithbbox, extract_sub_image


def create_crops(crop_method, extend_by_percent, data_json_path, crop_folder, image_folder_path):
    """Extract crops from images using entity bounding boxes."""
    # Set up paths
    image_folder_path = pathlib.Path(image_folder_path)

    # Determine output folder
    if crop_folder is None:
        suffix = f"{crop_method}_{extend_by_percent}" if extend_by_percent is not None else crop_method
        crop_folder_path = image_folder_path.parent.joinpath(f'crops_{suffix}')
    else:
        crop_folder = pathlib.Path(crop_folder)
        suffix = f"{crop_method}_{extend_by_percent}" if extend_by_percent is not None else crop_method
        crop_folder_path = crop_folder.joinpath(f'crops_{suffix}')

    # Create output directory
    crop_folder_path.mkdir(parents=True, exist_ok=True)
    print(f'##### Saving crops to {crop_folder_path} #####')

    # Load annotations
    with open(pathlib.Path(data_json_path), 'r') as file:
        data = json.load(file)

    # Extract crops for each annotation
    for idx, annot in tqdm(data.items()):
        # Copy original image
        image_name = annot['img_filename']
        image_path = image_folder_path.joinpath(image_name)
        copy_image_path = crop_folder_path.joinpath(image_name)
        shutil.copy2(image_path, copy_image_path)

        # Load original image
        image = Image.open(image_path)
        width, height = image.size

        # Process each entity
        entities = annot.get('relevant_entities', [])
        for i in range(len(entities)):
            entity_dict = entities[i]
            for entity_name, entity_bb in entity_dict.items():
                # Apply appropriate cropping method
                if crop_method == 'visual_cot':
                    normalized_bbox = normalize_bbox(
                        xywh_to_xyxy(entity_bb),
                        width,
                        height
                    )
                    crop = cropwithbbox(image, normalized_bbox)
                elif crop_method == 'normal':
                    crop = extract_sub_image(image, entity_bb, extend_by_percent)

                # Save crop
                crop_name = f'{image_name.split(".")[0]}_{i}.jpg'
                crop_path = crop_folder_path.joinpath(crop_name)
                crop.save(crop_path)

    print(f'##### All crops saved to {crop_folder_path} #####')


def main():
    """
    Main function to handle argument parsing and function execution.
    """
    parser = argparse.ArgumentParser(description="Convert JSON to LLaVA-compatible format.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the image folder.")
    parser.add_argument('--prompt_format', type=str, default="QA", help="Prompt format to use.")
    parser.add_argument('--output_dir', type=str, required=True, help="Parent directory to save the image crops.")
    parser.add_argument('--crop_method', type=str, required=True, default='normal', choices=['normal', 'visual_cot'])
    parser.add_argument('--extend_by_percent', type=int, default=None, help='Only works with normal')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    create_crops(
        crop_method=args.crop_method,
        extend_by_percent=args.extend_by_percent,
        data_json_path=args.input,
        crop_folder=args.output_dir,
        image_folder_path=args.image_folder
    )

if __name__ == "__main__":
    main()
