import json
import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional, Any
from tqdm import tqdm
from llava.riv_cot_utils import format_coco_to_normalized_bbox

def lower_first_letter(s: str) -> str:
    """
    Convert only the first letter of the given string to lowercase.

    Parameters:
    - s (str): The input string.

    Returns:
    - str: The string with first letter lowercased.
    """
    if not s:
        return s
    return s[0].lower() + s[1:]


def get_possible_answers(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract possible answers from the sample data.

    Parameters:
    - sample (dict): The sample data containing possible answers.

    Returns:
    - tuple: A tuple containing possible answers for question 1 and question 2 (if applicable).
    """
    possible_answers = sample["possible_answers"]

    if sample["has_multiple_questions"]:
        answers_between_first_and_second_question = ['A', 'B']

        possible_answers_1_list = []
        possible_answers_2_list = []

        for key, value in possible_answers.items():
            answer_text = f"({key}) {value}"
            if key in answers_between_first_and_second_question:
                possible_answers_1_list.append(answer_text)
            else:
                possible_answers_2_list.append(answer_text)

        return " ".join(possible_answers_1_list), " ".join(possible_answers_2_list)
    else:
        possible_answers_list = [f"({key}) {value}" for key, value in possible_answers.items()]
        return " ".join(possible_answers_list), ""


def format_entities_text(sample: Dict[str, Any], include_bbox: bool = False,
                         include_ids: bool = False, include_image_tag: bool = False) -> str:
    """
    Format entity information for inclusion in the prompt.

    Parameters:
    - sample (dict): The sample data containing entities.
    - include_bbox (bool): Whether to include bounding box information.
    - include_ids (bool): Whether to include entity IDs.
    - include_image_tag (bool): Whether to include image tags.

    Returns:
    - str: Formatted entity text.
    """
    entities = sample.get('relevant_entities', [])

    if not entities:
        return ""

    entity_strings = []
    for i, entity in enumerate(entities):
        name, bbox = list(entity.items())[0]
        entity_str = name

        if include_ids:
            entity_str += f" <id_{i + 1}>"

        if include_image_tag:
            entity_str += " <image>"

        if include_bbox:
            normalized_coords = format_coco_to_normalized_bbox(bbox, sample['img_size'][0], sample['img_size'][1])
            entity_str += f" [{normalized_coords[0]:.3f}, {normalized_coords[1]:.3f}, {normalized_coords[2]:.3f}, {normalized_coords[3]:.3f}]"

        entity_strings.append(entity_str)

    return "The relevant entities for this problem are: " + ", ".join(entity_strings) + "."


def process_explanation(explanation: str, sample: Dict[str, Any], include_bbox: bool = False) -> str:
    """
    Process explanation text to handle formatting of entity references.

    Parameters:
    - explanation (str): The explanation text to process.
    - sample (dict): The sample data.
    - include_bbox (bool): Whether to include bounding box information.
    - include_ids (bool): Whether to include entity IDs.

    Returns:
    - str: Processed explanation text.
    """
    modified_explanation = explanation

    for i, entity in enumerate(sample.get('relevant_entities', [])):
        name, bbox = list(entity.items())[0]
        normalized_coords = format_coco_to_normalized_bbox(bbox, *sample['img_size'])
        normalized_bbox = f"[{normalized_coords[0]:.3f}, {normalized_coords[1]:.3f}, {normalized_coords[2]:.3f}, {normalized_coords[3]:.3f}]"

        # Replace bold text
        pattern_bold = r'\*\*' + re.escape(name) + r'\*\*'
        modified_explanation = re.sub(pattern_bold, name, modified_explanation)

        # Replace bounding box
        pattern_bbox = re.escape(name) + r'\s\[\d+(\.\d+)?,\s\d+(\.\d+)?,\s\d+(\.\d+)?,\s\d+(\.\d+)?\]'

        if include_bbox:
            modified_explanation = re.sub(pattern_bbox, f"{name} {normalized_bbox}", modified_explanation)
        else:
            modified_explanation = re.sub(pattern_bbox, name, modified_explanation)

    return modified_explanation


def get_prompt_components(sample: Dict[str, Any], dataset: str, explanation_type: str) -> Dict[str, str]:
    """
    Get various prompt components based on the sample and explanation type.

    Parameters:
    - sample (dict): The sample data.
    - dataset (str): The dataset considered.
    - explanation_type (str): Type of explanation to use.

    Returns:
    - dict: Dictionary containing prompt components.
    """
    # Main instruction
    main_instruction_prompt = "Select all correct answers to the following question from the available options. "

    # Extract possible answers for the questions
    possible_answers_question_1, possible_answers_question_2 = get_possible_answers(sample)

    # Prepare answer instruction
    answer_only_prompt = "Provide the letters corresponding to your answer in the format: 'Answer(s): <letters>'."

    if dataset== "drivingvqa":
        context_prompt = f"Unless explicitly stated otherwise, assume you are driving a {sample['exam_type']} in France.\n"
        explanation_answer_prompt = "Detail your reasoning step by step based on road signs, markings, signals, and relevant driving rules. "
        explanation_following_entities = "Detail your reasoning step by step based on these entities and relevant driving rules. "
        entities_prompt = "List all relevant entities from the scene that are necessary to answer the following question, such as road signs, markings, signals, or other vehicles in the image. "
        entities_bbox_prompt = "List all relevant entities from the scene that are necessary to answer the following question, such as road signs, markings, signals, or other vehicles in the image, along with their bounding boxes."
        interleaved_explanation_bbox_answer_prompt = "Detail your reasoning step by step based on road signs, markings, signals, and relevant driving rules, including the bounding box next to the relevant entity. "
        interleaved_explanation_visual_answer_prompt = "Detail your reasoning step by step based on road signs, markings, signals, and relevant driving rules. In your reasoning, include the id referring next to the relevant entity. "
    elif dataset== "aokvqa":
        context_prompt = ""
        explanation_answer_prompt = "Detail your reasoning step by step. "
        explanation_following_entities = "Detail your reasoning step by step based on these entities."
        entities_prompt = "List all relevant entities from the scene that are necessary to answer the following question."
        entities_bbox_prompt = "List all relevant entities from the scene that are necessary to answer the following question, along with their bounding boxes."
        interleaved_explanation_bbox_answer_prompt = "Detail your reasoning step by step based on the bounding box next to the relevant entity. "
        interleaved_explanation_visual_answer_prompt = ""
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")


    # Format question text
    if sample['has_multiple_questions']:
        main_instruction_prompt = main_instruction_prompt.replace("question", "questions")
        entities_prompt = entities_prompt.replace("question", "questions")
        entities_bbox_prompt = entities_bbox_prompt.replace("question", "questions")
        main_instruction_prompt += " Choose at least one answer per question."
        q_text = (f"Question: {sample['questions'][0]}\nOptions: {possible_answers_question_1}.\n"
                  f"Question: {sample['questions'][1]}\nOptions: {possible_answers_question_2}.")
    else:
        q_text = f"Question: {sample['questions']}\nOptions: {possible_answers_question_1}."

    # Get explanation based on type
    if explanation_type == "original":
        explanation = sample['explanation']
    elif explanation_type == "interleaved":
        explanation = sample['interleaved_explanation']

    # Process explanations
    standard_explanation = process_explanation(explanation, sample)
    interleaved_bbox_explanation = process_explanation(sample['interleaved_explanation'], sample, include_bbox=True)

    return {
        'context_prompt': context_prompt,
        'main_instruction_prompt': main_instruction_prompt,
        'q_text': q_text,
        'answers_text': ", ".join(sample['true_answers']) + ".",
        'answer_only_prompt': answer_only_prompt,
        'explanation_answer_prompt': explanation_answer_prompt,
        'explanation_following_entities': explanation_following_entities,
        'entities_prompt': entities_prompt,
        'entities_bbox_prompt': entities_bbox_prompt,
        'interleaved_explanation_bbox_answer_prompt': interleaved_explanation_bbox_answer_prompt,
        'interleaved_explanation_visual_answer_prompt': interleaved_explanation_visual_answer_prompt,
        'explanation_text': f"Reasoning: {standard_explanation}",
        'interleaved_explanation_bbox_text': f"Reasoning: {interleaved_bbox_explanation}",
    }

def build_prompt(sample: Dict[str, Any], dataset: str, explanation_type: str,
                 prompt_format: str) -> Tuple[str, str, Optional[Union[str, List[str]]], Optional[Any]]:
    """
    Build the prompt and expected output based on the specified format.

    Parameters:
    - sample (dict): The sample data.
    - dataset (str): The dataset considered.
    - explanation_type (str): Type of explanation to use.
    - prompt_format (str): Format of the prompt.

    Returns:
    - tuple: (input_prompt, output, human_entities_prompt, final_output)
    """
    prompt_components = get_prompt_components(sample, dataset, explanation_type)

    # Initialize return values
    human_entities_prompt = None
    final_output = None

    # Handle different prompt formats
    if prompt_format == "QP-A":
        instruction = prompt_components['context_prompt'] + prompt_components['main_instruction_prompt'] + \
                      prompt_components['answer_only_prompt']
        input_prompt = f"{instruction}\n{prompt_components['q_text']}".strip()
        output = f"Answer(s): {prompt_components['answers_text']}".strip()

    elif prompt_format == "QP-EA":
        instruction = (prompt_components['context_prompt'] + prompt_components['main_instruction_prompt'] +
                       prompt_components['explanation_answer_prompt'] + 'Then, ' +
                       lower_first_letter(prompt_components['answer_only_prompt']))
        input_prompt = f"{instruction}\n{prompt_components['q_text']}".strip()
        output = f"{prompt_components['explanation_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()

    elif prompt_format == "QP-REA":
        instruction = (prompt_components['context_prompt'] + prompt_components['entities_prompt'] + "Then, " +
                       lower_first_letter(prompt_components['main_instruction_prompt']) +
                       prompt_components['explanation_following_entities'] + prompt_components['answer_only_prompt'])
        entities_text = format_entities_text(sample)
        input_prompt = f"{instruction}\n{prompt_components['q_text']}".strip()
        output = f"{entities_text}\n{prompt_components['explanation_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()

    elif prompt_format == "QP-RBEA":
        instruction = (prompt_components['context_prompt'] + prompt_components['entities_bbox_prompt'] + "Then, " +
                       lower_first_letter(prompt_components['main_instruction_prompt']) +
                       prompt_components['explanation_following_entities'] + prompt_components['answer_only_prompt'])
        entities_text = format_entities_text(sample, include_bbox=True)
        input_prompt = f"{instruction}\n{prompt_components['q_text']}".strip()
        output = f"{entities_text}\n{prompt_components['explanation_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()

    elif prompt_format == "QP-RB-RV-EA":
        instruction = prompt_components['context_prompt'] + prompt_components['entities_prompt']
        entities_text = format_entities_text(sample, include_bbox=True)

        if sample.get('relevant_entities'):
            human_entities_text = format_entities_text(sample, include_image_tag=True)
            human_entities_text = human_entities_text.replace("The relevant entities for this problem are",
                                                              "Their corresponding image patches are")
            human_entities_prompt = (human_entities_text + "Then, " +
                                     lower_first_letter(prompt_components['main_instruction_prompt']) +
                                     prompt_components['explanation_following_entities'] +
                                     prompt_components['answer_only_prompt'])
        else:
            human_entities_prompt = ("Then, " +
                                     lower_first_letter(prompt_components['main_instruction_prompt']) +
                                     prompt_components['explanation_following_entities'] +
                                     prompt_components['answer_only_prompt'])

        input_prompt = f"{instruction}\n{prompt_components['q_text']}".strip()
        output = f"{entities_text}".strip()
        final_output = f"{prompt_components['explanation_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()

    elif prompt_format == "QP-IEA":
        instruction = (prompt_components['context_prompt'] + prompt_components['main_instruction_prompt'] +
                       prompt_components['explanation_answer_prompt'] + 'Then, ' +
                       lower_first_letter(prompt_components['answer_only_prompt']))
        input_prompt = f"{instruction}\n{prompt_components['q_text']}".strip()
        output = f"{prompt_components['explanation_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()

    elif prompt_format == "QP-IBEA":
        instruction = (prompt_components['context_prompt'] + prompt_components['main_instruction_prompt'] +
                       prompt_components['interleaved_explanation_bbox_answer_prompt'] + 'Then, ' +
                       lower_first_letter(prompt_components['answer_only_prompt']))
        input_prompt = f"{instruction}\n{prompt_components['q_text']}".strip()
        output = f"{prompt_components['interleaved_explanation_bbox_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()

    elif prompt_format == "RIV-COT":
        instruction = (prompt_components['context_prompt'] + prompt_components['main_instruction_prompt'] +
                       prompt_components['interleaved_explanation_bbox_answer_prompt'] + 'Then, ' +
                       lower_first_letter(prompt_components['answer_only_prompt']))
        input_prompt = f"{instruction}\n{prompt_components['q_text']}".strip()

        # Split the explanation text at the bounding box coordinates
        if sample.get('relevant_entities'):
            explanation_parts = re.split(r'(\[[-]?\d+\.\d+,\s*[-]?\d+\.\d+,\s*[-]?\d+\.\d+,\s*[-]?\d+\.\d+\])',
                                         prompt_components['interleaved_explanation_bbox_text'])
            output = f"{explanation_parts[0]}{explanation_parts[1]}".strip()
            final_output = explanation_parts[2:]
            final_output[-1] += f"\nAnswer(s): {prompt_components['answers_text']}"
            human_entities_prompt = [" <image>"] * math.ceil(len(final_output) / 2)
        else:
            output = f"{prompt_components['interleaved_explanation_bbox_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()
            human_entities_prompt, final_output = None, None

    elif prompt_format == "QPR-EA":
        instruction = (prompt_components['context_prompt'] + prompt_components['main_instruction_prompt'] +
                       prompt_components['explanation_answer_prompt'] + 'Then, ' +
                       lower_first_letter(prompt_components['answer_only_prompt']))
        entities_text = format_entities_text(sample)
        input_prompt = f"{instruction}\n{prompt_components['q_text']}\n{entities_text}".strip()
        output = f"{prompt_components['explanation_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()

    elif prompt_format == "QPRB-EA":
        instruction = (prompt_components['context_prompt'] + prompt_components['main_instruction_prompt'] +
                       prompt_components['explanation_answer_prompt'] + 'Then, ' +
                       lower_first_letter(prompt_components['answer_only_prompt']))
        entities_text = format_entities_text(sample, include_bbox=True)
        input_prompt = f"{instruction}\n{prompt_components['q_text']}\n{entities_text}".strip()
        output = f"{prompt_components['explanation_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()

    elif prompt_format == "QPRV-EA":
        instruction = (prompt_components['context_prompt'] + prompt_components['main_instruction_prompt'] +
                       prompt_components['explanation_answer_prompt'] + 'Then, ' +
                       lower_first_letter(prompt_components['answer_only_prompt']))
        entities_text = format_entities_text(sample, include_image_tag=True)
        input_prompt = f"{instruction}\n{prompt_components['q_text']}\n{entities_text}".strip()
        output = f"{prompt_components['explanation_text']}\nAnswer(s): {prompt_components['answers_text']}".strip()

    return input_prompt, output, human_entities_prompt, final_output


def get_image_dict(image_folder: str) -> Dict[str, List[str]]:
    """
    Create a dictionary mapping base image names to their full filenames.

    Parameters:
    - image_folder (str): Path to the folder containing images.

    Returns:
    - dict: Dictionary mapping base names to image filenames.
    """
    image_dict = defaultdict(list)

    for img_filename in os.listdir(image_folder):
        basename = img_filename.split('.')[0].split('_')[0]
        image_dict[basename].append(img_filename)

    # Sort each list to ensure the order: main file, then _0, _1, etc.
    for key in image_dict:
        image_dict[key] = sorted(image_dict[key], key=lambda x: (x.count('_'), x))

    return image_dict


def convert_to_llava_format(input_file: str, dataset: str, image_folder: str, prompt_format: str,
                            explanation_type: str, output_dir: str, no_image: bool = False) -> None:
    """
    Convert the input JSON data to LLaVA-compatible format and save it.

    Parameters:
    - input_file (str): Path to the input JSON file.
    - dataset (str): The dataset considered.
    - image_folder (str): Path to the folder containing images.
    - prompt_format (str): Format of the prompt.
    - explanation_type (str): Type of explanation to use.
    - output_dir (str): Directory to save the LLaVA-compatible JSON.
    - no_image (bool): Whether to exclude images from the prompt.
    """

    # Load the input JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    llava_format = []

    if no_image:
        print(f'No image: {no_image}')

    # Create a dictionary of image filenames
    image_dict = get_image_dict(image_folder) if not no_image else {}

    for sample_id, sample in tqdm(data.items(), desc="Converting samples"):
        # Build prompt input and output
        input_prompt, output_response, opt_human_prompt, opt_final_output = build_prompt(
            sample, dataset, explanation_type, prompt_format
        )

        if not no_image:
            input_prompt = f"<image>\n{input_prompt}"
            # Get all the image filenames in the image folder
            if "V" in prompt_format:
                img_list = image_dict.get(sample['img_filename'].split('.')[0], [sample['img_filename']])
            else:
                img_list = [sample['img_filename']]

        # Create the conversation entries
        conversations = [
            {
                "from": "human",
                "value": input_prompt
            },
            {
                "from": "gpt",
                "value": output_response
            }
        ]

        # Handle optional additional conversation turns
        if isinstance(opt_human_prompt, str) and opt_human_prompt:
            conversations.append({
                "from": "human",
                "value": opt_human_prompt
            })
            conversations.append({
                "from": "gpt",
                "value": opt_final_output
            })
        elif isinstance(opt_human_prompt, list):
            for i, prompt in enumerate(opt_human_prompt):
                conversations.append({
                    "from": "human",
                    "value": prompt
                })
                output_value = (f"{opt_final_output[2 * i]}{opt_final_output[2 * i + 1]}"
                                if 2 * i + 1 < len(opt_final_output)
                                else opt_final_output[2 * i])
                conversations.append({
                    "from": "gpt",
                    "value": output_value
                })

        # Append the entry to the LLaVA format
        if not no_image:
            entry = {"id": sample_id, "image": img_list, "conversations": conversations}
        else:
            entry = {"id": sample_id, "conversations": conversations}

        llava_format.append(entry)

    # Output the LLaVA-compatible JSON
    subset_split = os.path.basename(input_file).split('.')[0]

    if no_image:
        output_file = os.path.join(output_dir,
                                   f"llava-{subset_split}-{explanation_type}-{prompt_format.lower()}-no-img.json")
    else:
        output_file = os.path.join(output_dir,
                                   f"llava-{subset_split}-{explanation_type}-{prompt_format.lower()}-{os.path.basename(image_folder)}.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(llava_format, f, ensure_ascii=False, indent=4)
    print(f"LLaVA format data saved to: {output_file}")


def main():
    """
    Main function to handle argument parsing and function execution.
    """
    parser = argparse.ArgumentParser(description="Convert JSON to LLaVA-compatible format.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--dataset',type=str, required=True, help="Chose between DrivingVQA or AOKVQA dataset.")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the image folder.")
    parser.add_argument('--prompt_format', type=str, default="QA", help="Prompt format to use.")
    parser.add_argument('--explanation_type', type=str, default="interleaved", help="Explanation type to use.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the LLaVA-compatible JSON.")
    parser.add_argument('--no_image', action='store_true', help="Flag to exclude images from the prompt.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Convert to LLaVA format
    convert_to_llava_format(
        input_file=args.input,
        dataset=args.dataset,
        image_folder=args.image_folder,
        prompt_format=args.prompt_format,
        explanation_type=args.explanation_type,
        output_dir=args.output_dir,
        no_image=args.no_image,
    )

if __name__ == "__main__":
    main()

