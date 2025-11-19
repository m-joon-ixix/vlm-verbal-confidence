import random
import argparse
from tqdm import tqdm

from common.random_utils import get_seed
from common.json_utils import load_from_json, dump_to_json
from common.const import FLAG_TO_SPLIT, DATASETS
from method.r_tuning_prefixes import *
from method.utils import count_correct_answers
from generate.form_query import (
    build_finetune_user_content,
    build_finetune_assistant_content,
)


def main(args):
    random.seed(get_seed())
    dataset = []

    with open(f"instruction/generate_description.txt", encoding="utf-8") as f:
        image_desc_instruction = "".join(f.readlines())

    prefix_boundary_dict = _get_prefix_boundaries(args)
    visual_boundary_alpha, visual_boundary_beta = prefix_boundary_dict["visual"]
    answer_boundary_alpha, answer_boundary_beta = prefix_boundary_dict["answer"]

    used_gt_desc_count = 0

    data_list = load_from_json(
        f"./output/{args.dataset_name}/response/{args.model_key}/{_output_filename(args)}"
    )
    for data in tqdm(data_list, desc="Constructing Training Data (Two Stage)"):
        # visual prefix
        if data["image_desc_confidence"] < visual_boundary_alpha:
            visual_prefix = random.choice(VISUAL_UNKNOWN_PREFIXES)
        elif data["image_desc_confidence"] < visual_boundary_beta:
            visual_prefix = random.choice(VISUAL_UNCERTAIN_PREFIXES)
        else:
            visual_prefix = random.choice(VISUAL_CONFIDENT_PREFIXES)

        visual_response = f"{visual_prefix} {random.choice(data['sample_image_desc'])}"

        # answer prefix
        correct_count = count_correct_answers(data)[0]

        # NOTE: If Visual confidence is UNKNOWN or UNCERTAIN, and Answer confidence is CONFIDENT => replace `visual_response` with CONFIDENT prefix + GT desc
        if (
            data["image_desc_confidence"] < visual_boundary_beta
            and correct_count >= answer_boundary_beta
        ):
            visual_prefix = random.choice(VISUAL_CONFIDENT_PREFIXES)
            visual_response = f"{visual_prefix} {data['extracted_image_desc_gt']}"
            used_gt_desc_count += 1

        if correct_count < answer_boundary_alpha:
            prefix = random.choice(ANSWER_UNKNOWN_PREFIXES)
        elif correct_count < answer_boundary_beta:
            prefix = random.choice(ANSWER_UNCERTAIN_PREFIXES)
        else:
            prefix = random.choice(ANSWER_CONFIDENT_PREFIXES)

        selected_response_idx = random.randint(0, len(data["responses"]) - 1)
        selected_response = data["responses"][selected_response_idx]
        new_response = f"{prefix} {selected_response}"

        first_round_messages = [
            {
                "role": "user",
                "content": build_finetune_user_content(
                    f"### Question:\n{data['question']}\n{image_desc_instruction}",
                    args.model_key,
                ),
            },
            {
                "role": "assistant",
                "content": build_finetune_assistant_content(visual_response),
            },
        ]

        second_round_messages = [
            {
                "role": "user",
                "content": build_finetune_user_content(
                    data["queries"][selected_response_idx], args.model_key
                ),
            },
            {
                "role": "assistant",
                "content": build_finetune_assistant_content(new_response),
            },
        ]

        messages = first_round_messages + second_round_messages
        image_path = f"./output/{args.dataset_name}/dataset/{FLAG_TO_SPLIT[args.split_flag]}_image/{data['id']}.png"
        dataset.append({"messages": messages, "image_path": image_path})

    print(f"[INFO] Image desc was replaced with GT in {used_gt_desc_count} examples.")

    output_file = f"./output/{args.dataset_name}/training_data/{args.model_key}/better_imgconf_corr/{_output_filename(args)}"
    dump_to_json(output_file, dataset)


def _get_prefix_boundaries(args):
    return load_from_json(
        f"./output/{args.dataset_name}/training_data/{args.model_key}/distribution/{_output_filename(args)}"
    )["boundaries"]


def _output_filename(args):
    return f"rtuning_base_{FLAG_TO_SPLIT[args.split_flag]}.json"


# PYTHONPATH=. python dataset/construct_train_set/replace_lowconf_desc_to_gt.py --dataset-name MMMU --model-key qwen --split-flag t
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-name", type=str, choices=DATASETS)
    parser.add_argument("--model-key", type=str, choices=["qwen", "llama", "llava"])
    parser.add_argument("--split-flag", type=str, choices=["t", "v"])

    args = parser.parse_args()
    main(args)
