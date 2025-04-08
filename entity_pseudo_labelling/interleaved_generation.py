import os
import json
import argparse
import re
import base64
from tqdm import tqdm
from openai import OpenAI

# Replace with your OpenAI API key
OPENAI_API_KEY = ''
client = OpenAI(api_key=os.environ["OPENAIKEY"])


def format_instruction(input_sample):
    true_answers = input_sample['true_answers']
    question = input_sample['questions']
    options = input_sample['possible_answers']
    answer = [options[i] for i in true_answers]
    explanation = input_sample['explanation']
    annotations_samples = input_sample['relevant_entities']

    instructions = f"""
    You are given an image of the driving theory test, a question about this image, a list of options, and the reasoning leading to the correct answer.
    I also give you a lists of entities along with bounding box coordinates, that are relevant to answering the question.
    I need you to:
    1. Convert the original reasoning into a clear, step-by-step reasoning that makes use of all entities in the list.
    2. If the original reasoning does not mention one of the entities at all, add a short sentence referencing that entity and link it with the reasonign steps.
    3. Refer to each relevant entity in the format: **entity_name** [x1, x2, y1, y2]. Replicate bounding box coordinates exactly as provided in the list.
    4. Keep all other wording as close to the original reasoning as possible.
    5. The entities should always be mentioned at the beginning of the sentences.
    **Question**: {question}
    **Options**: {options}
    **Answers**: {answer}
    **Reasoning**: {explanation}
    **Entities**: {annotations_samples}
    **Interleaved Reasoning**:
    """

    return instructions


def interleaved_generation_via_gpt(sample, demo_samples, image_folder, model="gpt-4o-mini-2024-07-18"):
    system = {
        "role": "system",
        "content": "You are an expert at driving theory. You are tasked with helping a student answering questions about driving scenes."}
    messages = [system]

    if len(sample['relevant_entities']) == 0:
        return sample['explanation']

    # Add demo samples
    for demo_sample in demo_samples.values():
        with open(os.path.join(image_folder, demo_sample['img_filename']), "rb") as image_file:
            demo_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{demo_image_base64}"}},
                {"type": "text", "text": format_instruction(demo_sample)}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": demo_sample["interleaved_explanation"]
        })

    # Add sample
    with open(os.path.join(image_folder, sample['img_filename']), "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": format_instruction(sample)}
        ]
    })

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content


def generate_interleaved_reasoning(input_path, output_path, image_folder, model, save_every):
    """
    Processes dataset and generates interleaved explanations.

    Args:
        input_path (str): Path to input JSON.
        output_path (str): Path to output annotated JSON.
        image_folder (str): Directory containing images.
        model (str): GPT model name.
        save_every (int): Number of samples after which progress should be saved.
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Demo samples for few-shot prompting for DrivingVQA, to replace for your own dataset
    demo_samples = {'2328': data['2328'], '3414': data['3414']}
    demo_samples['2328']["interleaved_explanation"] = (
        "A **deceleration lane** [933.62, 522.54, 537.35, 249.17] allows me to exit without disrupting traffic. "
        "An **exit sign** [904.6, 413.47, 57.03, 36.02] indicates the exit. "
        "My **rear-view mirror** [952.52, 82.74, 544.27, 181.17] shows a **vehicle** [1206.9, 156.54, 44.8, 28.56] far behind, so I can slow down."
    )
    demo_samples['3414']["interleaved_explanation"] = (
        "The **bicycle sign** [766.38, 52.65, 36.16, 39.36] shows cyclists may come in the opposite direction. "
        "The **speed limit sign** [763.17, 5.96, 40.74, 48.98] indicates a 30 km/h speed limit zone."
    )

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

        # Generate interleaved explanation for the sample
        explanation = interleaved_generation_via_gpt(sample, demo_samples, image_folder, model)
        sample["interleaved_explanation"] = explanation
        processed_samples[sample_id] = sample

        # Save periodically
        if i % save_every == 0:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(processed_samples, output_file, ensure_ascii=False, indent=4)

    print(f"Interleaved data saved at: {output_path}")


def clean_interleaved_explanation(output_path):
    """Clean and validate interleaved explanations."""
    cleaned_samples = {}

    with open(output_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Demo samples for few-shot prompting for DrivingVQA, to replace for your own dataset
    demo_samples = {'2328': data['2328'], '3414': data['3414']}
    demo_samples['2328']["interleaved_explanation"] = (
        "A **deceleration lane** [933.62, 522.54, 537.35, 249.17] allows me to exit without disrupting traffic. "
        "An **exit sign** [904.6, 413.47, 57.03, 36.02] indicates the exit. "
        "My **rear-view mirror** [952.52, 82.74, 544.27, 181.17] shows a **vehicle** [1206.9, 156.54, 44.8, 28.56] far behind, so I can slow down."
    )
    demo_samples['3414']["interleaved_explanation"] = (
        "The **bicycle sign** [766.38, 52.65, 36.16, 39.36] shows cyclists may come in the opposite direction. "
        "The **speed limit sign** [763.17, 5.96, 40.74, 48.98] indicates a 30 km/h speed limit zone."
    )
    demo_bbs = [list(ent.values())[0] for sample in demo_samples.values() for ent in sample['relevant_entities']]

    for (sample_id, sample) in tqdm(data.items()):
        old_entities = [list(item.keys())[0] for item in sample['relevant_entities']]
        all_bbs = [list(item.values())[0] for item in sample['relevant_entities']]
        result = sample["interleaved_explanation"]

        # Check if the sample is a demo sample
        if sample_id in demo_samples.keys():
            sample["updated_relevant_entities"] = sample['relevant_entities']
            sample["interleaved_explanation_cleaned"] = result

        # deal with: cases when an entity has ** around but no bounding box coordinates
        # Find instances of **text** and checks if they are followed by coordinates. If not, it removes the **
        pattern = r'\*\*(.*?)\*\*(\s*\[\s*-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?\])?'
        result = re.sub(pattern, lambda m: m.group(1) if m.group(2) is None else f"**{m.group(1)}**{m.group(2)}", result)

        # Extract new dict of entities, and clean interleaved explanation and entities list
        matches = re.findall(r'\*\*(.*?)\*\*\s*\[(.*?)\]', result)
        entities = []
        for entity_name, coords_text in matches:
            try:
                # Attempt to parse the coordinates as floats
                coords = list(map(float, coords_text.split(', ')))
                entity = {entity_name: coords}
                # remove entity if coordinates aren't in the original list of relevant entities
                if coords not in all_bbs:
                    # Check if some coordinates are the same as in the demonstrations
                    if coords in demo_bbs and entity_name in old_entities:
                        # Then the coordinates are wrong, but the entity is correct: update interleaved explanation:
                        correct_coords = [list(item.values())[0] for item in sample['relevant_entities'] if list(item.keys())[0] == entity_name][0]
                        result = result.replace(f"**{entity_name}** [{coords_text}]", f"**{entity_name}** {correct_coords}")
                        entities.append({entity_name: correct_coords})
                    else:
                        # Then the entity is wrong: update interleaved explanation: remove coordinates associated with the entity, and don't add the entity to new list.
                        result = result.replace(f"**{entity_name}** [{coords_text}]", entity_name)
                # remove duplicated entities
                elif entity not in entities:
                    entities.append(entity)
                else:
                    # update interleaved explanation: remove duplicated occurrence of entity
                    result = result.replace(f"**{entity_name}** [{coords_text}]", entity_name, 1)
            except ValueError:
                # Skip this entry if coordinates cannot be converted to floats (e.g. the model puts a string instead)
                result = result.replace(f"**{entity_name}** [{coords_text}]", entity_name, 1)

        # check if a bb is repeated for two different entity labels:
        entities_bbs = [list(item.values())[0] for item in entities]
        duplicated_bbs = list({tuple(item) for item in entities_bbs if entities_bbs.count(item) > 1})
        # if so, check if one of them is in the original relevant_entities
        if len(duplicated_bbs)==1:
            duplicated_entities = [entity for entity in entities if list(entity.values())[0] == list(duplicated_bbs[0])]
            # if we have duplicates, keep only the correct one (if there is) and remove the others from the list of entities
            correct_entity = [entity for entity in [list(item.keys())[0] for item in entities] if entity in old_entities]
            if correct_entity:
                wrong_entities = [entity for entity in duplicated_entities if list(entity.keys())[0] != correct_entity[0]]
                for wrong_entity in wrong_entities:
                    # remove from list of entities
                    entities = [entity for entity in entities if entity != wrong_entity]
                    # remove from interleaved explanation
                    result = result.replace(f"**{list(wrong_entity.keys())[0]}** {list(duplicated_bbs[0])}", list(wrong_entity.keys())[0])

        # case where there are two relevant_entities with same label but different BBs, and the model only found one in the explanation:
        duplicated_entities = set([x for x in old_entities if old_entities.count(x) > 1])
        if len(entities) != len(sample['relevant_entities']) and len(duplicated_entities)>0 and len(entities)>0:
            for entity in duplicated_entities:
                if entity in [list(item.keys())[0] for item in entities]:
                    entity_bb = [list(item.values())[0] for item in sample['relevant_entities'] if list(item.keys())[0] == entity]
                    # replace the unique bb in the interleaved explanation with the two bbs
                    found_bb = [list(item.values())[0] for item in entities if list(item.keys())[0] == entity][0]
                    result = result.replace(f"**{entity}** {found_bb}", f"**{entity}** {entity_bb[0]} and **{entity}** {entity_bb[1]}")
                    # add the missing entity to the list of entities
                    missing_entity = {entity: entity_bb[1]}
                    entities.append(missing_entity)

        sample["interleaved_explanation"] = result
        cleaned_samples[sample_id] = sample

    with open(output_path.replace('.json', '_cleaned.json'), 'w') as f:
        json.dump(cleaned_samples, f, indent=4)
    print(f"Cleaned explanations saved to: {output_path.replace('.json', '_cleaned.json')}")


def main():
    parser = argparse.ArgumentParser(description="Generate interleaved explanations for driving dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON dataset.")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument("--model_id", type=str, default="gpt-4o-mini-2024-07-18", help="GPT model to use.")
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()

    # Process the dataset and generate interleaved explanations
    generate_interleaved_reasoning(args.input, args.output, args.image_folder, args.model_id, args.save_every)

    # Clean the interleaved explanations
    clean_interleaved_explanation(args.output)

if __name__ == "__main__":
    main()
