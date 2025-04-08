import os
import json
import argparse
import base64
from tqdm import tqdm
from openai import OpenAI

# Replace with your OpenAI API key
OPENAI_API_KEY = ''
client = OpenAI(api_key=os.environ["OPENAIKEY"])

def get_possible_answers(sample):
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


def generate_system_prompt(dataset):
    """
    Generates a system prompt for ChatGPT based on the dataset type.

    Args:
        dataset (str): Either "drivingvqa" or "aokvqa".

    Returns:
        dict: System message for ChatGPT.
    """
    if dataset == "drivingvqa":
        system_text = (
            "You are a driving theory expert. Your role is to extract relevant entities from a driving scene, "
            "which will be passed to an open-set object detector for recognition."
        )
    else:
        system_text = (
            "You are an AI agent specializing in entity extraction. Your role is to extract relevant entities from a scene, "
            "which will be passed to an open-set object detector for recognition."
        )

    return {"role": "system", "content": system_text}


def generate_user_prompt(sample, dataset):
    """
    Generates a user prompt for entity extraction.

    Args:
        sample (dict): Sample data containing the question, options, and explanation.
        dataset (str): Either "AOKVQA" or "DrivingVQA".

    Returns:
        str: A formatted user prompt for entity extraction.
    """
    possible_answers_q1, possible_answers_q2 = get_possible_answers(sample)

    if sample.get('has_multiple_questions', False):
        problem_description = (
            f"Question 1: {sample['questions'][0]}\nOptions: {possible_answers_q1}.\n"
            f"Question 2: {sample['questions'][1]}\nOptions: {possible_answers_q2}."
        )
    else:
        problem_description = f"Question: {sample['questions']}\nOptions: {possible_answers_q1}."

    correct_answers = ", ".join(sample.get('true_answers', []))
    explanation = sample.get('explanation', "")

    if dataset=="drivingvqa":
        prompt = f"""
        **Instructions**
        Extract only the relevant entities visible in the scene that directly contribute to answering the problem below.
        Prioritize road signs, traffic control devices, vehicles, and road users.
        If entities are located in rear-view or side-view mirrors, include the mirror itself in the list.
        The output format should be a list of entities, such as ['cyclist', 'oncoming vehicle', 'solid line', 'pedestrian crossing'].

        **Problem**:
        "{problem_description}"

        **Correct answer(s)**: "{correct_answers}"
        **Explanation**: "{explanation}"
        """
    else:
        prompt = f"""
        **Instructions**
        Extract only the relevant entities visible in the scene that directly contribute to answering the problem below.
        Focus on objects or people in the scene and their relevant attributes.
        The output format should be a list of entities, such as ['man', 'table', 'window'].

        **Problem**:
        "{problem_description}"

        **Correct answer(s)**: "{correct_answers}"
        **Explanation**: "{explanation}"
        """

    return prompt.strip()


def infer_entities_via_gpt(sample, dataset, image_folder, model="gpt-4o-2024-08-06"):
    """
    Calls the OpenAI GPT API to extract relevant entities from an image and problem statement.

    Args:
        sample (dict): Sample data containing the problem, options, and explanation.
        dataset (str): Either "AOKVQA" or "DrivingVQA".
        image_folder (str): Path to the directory containing images.
        model (str): The OpenAI model to use.

    Returns:
        str: Extracted entities in a formatted response.
    """
    # Fetch image base64
    with open(os.path.join(image_folder, sample['img_filename']), "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        generate_system_prompt(dataset),
        {"role": "user", "content": generate_user_prompt(sample, dataset)},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content.strip()


# Function to process the dataset and add detected entities
def infer_entities(input_path, dataset, output_path, image_folder, save_every):
    """
    Infer entities using GPT on every samples of a given dataset.

    Args:
        input_path (str): Path to the input dataset JSON file.
        dataset (str): Either "AOKVQA" or "DrivingVQA".
        output_path (str): Path to the output JSON file.
        image_folder (str): Path to the directory containing images.
        save_every (int): Number of samples after which progress should be saved.
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Load previously processed results
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as output_file:
            processed_samples = json.load(output_file)
        last_processed_id = int(list(processed_samples.keys())[-1])
        print(f"Resuming from sample ID: {last_processed_id}")
    else:
        processed_samples = {}
        last_processed_id = -1
        print("Starting from the first sample")

    # Process each sample in the dataset
    for i, (sample_id, sample) in enumerate(tqdm(data.items())):
        if int(sample_id) <= last_processed_id:
            continue  # Skip already processed samples

        sample.pop('relevant_entities', None)  # Remove any existing entities
        sample.pop('interleaved_explanations', None)  # Remove corresponding interleaved explanations

        inferred_entities_str = infer_entities_via_gpt(sample, dataset, image_folder)

        # Parse the response as a list, fallback to empty list on failure
        try:
            inferred_entities = eval(inferred_entities_str)
        except (ValueError, SyntaxError):
            print(f"Error parsing inferred entities: {inferred_entities}")
            inferred_entities = []

        # Add a new entry to the sample
        sample['inferred_entities'] = inferred_entities
        processed_samples[sample_id] = sample

        # Save periodically
        if i % save_every == 0:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(processed_samples, output_file, ensure_ascii=False, indent=4)

    print(f"Entities have been inferred and saved. Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Entity Inference using ChatGPT")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["drivingvqa", "aokvqa"])
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()

    # Process the dataset
    infer_entities(
        input_path=args.input,
        dataset=args.dataset,
        output_path=args.output,
        image_folder=args.image_folder,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
