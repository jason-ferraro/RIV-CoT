import json
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
import re, ast, os
import glob
import base64
from collections import Counter
import argparse
from random import Random
import numpy as np

# Replace with your OpenAI API key
OPENAI_API_KEY = ''
client = OpenAI(api_key=os.environ["OPENAIKEY"])

PROMPT = """
        You are a strict but fair driving-theory instructor. You're given:
        1. A driving theory test question
        2. A list of possible answer options
        3. The official “correct reasoning”
        4. A student's reasoning for the same question.

        Your task: **Assess if the student's reasoning matches the correct reasoning.**

        ### Step-by-Step Instructions
        1. **Identify Student's Arguments**
        - List each key argument or step in the student's reasoning.
        - For each argument, briefly state whether it is correct or not, given the provided correct reasoning.

        2. **Check for Missing or Contradictory Points**
        - Look at the official correct reasoning. List **important points or steps** from the correct reasoning that the student **omits** or **directyl contradicts**.
        - Minor omissions or differences in wording/style are okay.

        3. **Decide on Overall Correctness**
        - If the student's reasoning is **mostly consistent** with the correct reasoning and has **no major factual errors** then it is considered “correct.”
        - If the student's reasoning **contains significant logical or factual errors**, or **omits critical steps** from the correct reasoning, then mark it “incorrect.”

        Important note:  The student's reasoning does not have to match the official reasoning exactly; it just needs to be conceptually equivalent and free of serious contradictions.

        ### Final Output Format
        - Provide your step by step analysis.
        - At the end, write:
        **Final Answer**: "1" if you judge the student's reasoning is overall correct, "0" if it is overall incorrect.
        """


def query_gpt(sample, model="gpt-4o-mini-2024-07-18", verbose=False, prompt=PROMPT):
    system = {
        "role": "system",
        "content": "You are an expert at driving theory. You are tasked with evaluating a student's answer to driving theory questions."}
    messages = [system]

    def format_instructions(input_sample):
        question = input_sample['questions']
        options = input_sample['possible_answers']
        explanation = input_sample['explanation']
        model_reasoning = input_sample['model_reasoning']

        instructions = prompt + f"""
        **Question**: {question}
        **Options**: {options}
        **Correct Reasoning**: {explanation}
        **Student's Reasoning**: {model_reasoning}
        """
        return instructions

    prompt = {
        "role": "user",
        "content": format_instructions(sample)
    }
    messages.append(prompt)

    if verbose:
        print(messages)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content


def clean_interleaved_explanation(text):
    # remove bb
    pattern = r"\[\d+(?:\.\d+)?, \d+(?:\.\d+)?, \d+(?:\.\d+)?, \d+(?:\.\d+)?\]"
    cleaned_text = re.sub(pattern, "", text)
    # remove starts
    cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned_text)
    # Remove extra spaces left after removal
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()
    return cleaned_text


def clean_model_reasoning(text):
    if "relevant entities for this problem are:" in text:
        pattern = r"^The relevant entities for this problem are:.*?\[\d+(?:\.\d+)?, \d+(?:\.\d+)?, \d+(?:\.\d+)?, \d+(?:\.\d+)?\]\.\s*"
        # Remove the matching sentence
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    # remove "Reasoning" and "Answer(s)"
    cleaned_text = re.sub(r"\bReasoning:\s*", "", text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bAnswer(?:\(s\))?:.*", "", cleaned_text, flags=re.IGNORECASE)

    # remove starts and boudning boxes
    cleaned_text = clean_interleaved_explanation(cleaned_text)

    # Remove any extra whitespace or newlines
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()
    return cleaned_text


def clean_answer(text):
    match = re.search(r"\*\*(?:Answer|Final Answer)\*\*:\s*\"?([01])\"?", text, flags=re.IGNORECASE)
    if not match and len(text) == 1:
        return text
    return match.group(1) if match else None


def clean_model_interleaved_reasoning(conv):
    """ Specific function to process RIV-CoT model output format.
    """
    text = [item['value'] for item in conv if item['from'] == 'gpt']
    text = ' '.join(text)
    cleaned_text = clean_model_reasoning(text)
    return cleaned_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folder', type=str, required=True,
                        help="Folder where outputs.json and predictions.json are saved")
    parser.add_argument('--test_json_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--judge_model', type=str, default='gpt-4o-mini')
    args = parser.parse_args()

    result_folder = args.result_folder
    output_path = args.output_path

    if not os.path.exists(result_folder):
        print('This prediction folder does not exist.')
        exit(0)

    if not os.path.exists(args.test_json_path):
        print('No test set file found')
        exit(0)
    with open(args.test_json_path, 'r') as f:
        test_set = json.load(f)

    print(f"Processing {result_folder}")
    print(f"Saving output in {output_path}")
    if os.path.exists(output_path):
        predictions_dict = json.load(open(output_path))
        print('Already done:', len(predictions_dict))
    else:
        predictions_dict = {}
    model_outputs = json.load(open(result_folder + 'outputs.json'))

    # run judge
    i = 0
    for sampleid, gt_sample in tqdm(test_set.items()):
        i += 1
        if sampleid in predictions_dict:
            continue
        new_sample = gt_sample.copy()
        model_sample = [s for s in model_outputs if s['id'] == sampleid][0]
        if 'riv-cot' in result_folder:
            new_sample['model_reasoning'] = clean_model_interleaved_reasoning(model_sample['conversations'])
        else:
            new_sample['model_reasoning'] = clean_model_reasoning(model_sample['conversations'][1]['value'])
        response = query_gpt(new_sample, args.judge_model)
        new_sample["gpt_reasoning_correctness"] = response
        predictions_dict[sampleid] = new_sample
        if i % 20 == 0:
            with open(output_path, 'w') as f:
                json.dump(predictions_dict, f, indent=4)
    with open(output_path, 'w') as f:
        json.dump(predictions_dict, f, indent=4)

    # extract performance
    performance_dict = {'exam_score': [], 'reasoning_correctness': []}

    predictions_dict = json.load(open(output_path))
    if len(predictions_dict) != len(test_set):
        print("Incomplete reasoning evaluation")
        exit(0)

    for id, sample in predictions_dict.items():
        sample["gpt_reasoning_correctness_clean"] = clean_answer(sample["gpt_reasoning_correctness"])
        if sample["gpt_reasoning_correctness_clean"] not in ["0", "1"]:
            print(sample["gpt_reasoning_correctness"])
            print(clean_answer(sample["gpt_reasoning_correctness"]))
            print("Error in parsing the judge output")
            exit(0)
    gpt_reasoning_correctness = [int(sample["gpt_reasoning_correctness_clean"]) for sample in predictions_dict.values()]
    reasoning_accuracy = sum(gpt_reasoning_correctness) / len(gpt_reasoning_correctness)
    performance_dict['reasoning_correctness'].append(reasoning_accuracy)
    # add score
    score = json.load(open(f"{result_folder}/scores.json"))['exam_score']
    performance_dict['exam_score'].append(score)

    print(performance_dict)
