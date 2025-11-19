import random
import argparse
from tqdm import tqdm
from typing import List

from common.random_utils import get_seed
from common.json_utils import load_from_json, dump_to_json
from common.const import FAILED_TOKEN, FLAG_TO_SPLIT, DATASETS, CONFIDENCE_LEVELS
from method.r_tuning_prefixes import *
from method.utils import count_correct_answers
from generate.form_query import (
    build_finetune_user_content,
    build_finetune_assistant_content,
)


def main(args):
    random.seed(get_seed())

    with open(f"instruction/generate_description.txt", encoding="utf-8") as f:
        image_desc_instruction = "".join(f.readlines())

    dataset = get_balanced_dataset(args.model_name, args.split_flag, args.n_each_area)
    formatted_dataset = [
        format_to_ft(data, args.model_name, image_desc_instruction)
        for data in tqdm(dataset, desc="Formatting balanced dataset")
    ]

    output_file = f"./output/all/training_data/{args.model_name}/balanced_each-{args.n_each_area}_{FLAG_TO_SPLIT[args.split_flag]}.json"
    dump_to_json(output_file, formatted_dataset)


def format_to_ft(data: dict, model_name: str, img_desc_inst: str) -> dict:
    # attach visual prefix
    visual_response = (
        f"{data['visual_prefix']} {random.choice(data['sample_image_desc'])}"
    )

    # attach answer prefix
    selected_response_idx = random.randint(0, len(data["responses"]) - 1)
    new_response = f"{data['answer_prefix']} {data['responses'][selected_response_idx]}"

    first_round_messages = [
        {
            "role": "user",
            "content": build_finetune_user_content(
                f"### Question:\n{data['question']}\n{img_desc_inst}",
                model_name,
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
                data["queries"][selected_response_idx], model_name
            ),
        },
        {
            "role": "assistant",
            "content": build_finetune_assistant_content(new_response),
        },
    ]

    messages = first_round_messages + second_round_messages
    image_path = f"./output/{data['dataset']}/dataset/{FLAG_TO_SPLIT[args.split_flag]}_image/{data['id']}.png"
    return {"messages": messages, "image_path": image_path}


def get_balanced_dataset(
    model_name: str, split_flag: str, n_each_area: int
) -> List[dict]:
    data_list = []
    # gather data from all datasets
    for dataset in DATASETS:
        _data_list = load_from_json(
            f"./output/{dataset}/response/{model_name}/rtuning_base_{FLAG_TO_SPLIT[split_flag]}.json",
            print_msg=False,
        )
        # mark the dataset name where each example is from
        for data in _data_list:
            data["dataset"] = dataset

        data_list += _data_list

    img_conf_levels = [f"img_{level}" for level in CONFIDENCE_LEVELS]
    ans_conf_levels = [f"ans_{level}" for level in CONFIDENCE_LEVELS]

    # ex) data_by_conf_level["img_unknown"]["ans_confident"]
    data_by_conf_level = {img_level: {} for img_level in img_conf_levels}
    for img_level in img_conf_levels:
        for ans_level in ans_conf_levels:
            data_by_conf_level[img_level][ans_level] = []  # List[dict]

    # group data by image/answer confidence levels
    for data in tqdm(data_list, desc="Categorizing examples on confidence"):
        assert data["image_desc_confidence"] != FAILED_TOKEN

        if data["image_desc_confidence"] < 6:  # 0-6
            img_level = f"img_{CONFIDENCE_LEVELS[0]}"
            visual_prefix = random.choice(VISUAL_UNKNOWN_PREFIXES)
        elif data["image_desc_confidence"] < 8:  # 6-8
            img_level = f"img_{CONFIDENCE_LEVELS[1]}"
            visual_prefix = random.choice(VISUAL_UNCERTAIN_PREFIXES)
        else:  # 8-10
            img_level = f"img_{CONFIDENCE_LEVELS[2]}"
            visual_prefix = random.choice(VISUAL_CONFIDENT_PREFIXES)

        correct_count, _ = count_correct_answers(data)
        if correct_count <= 0:  # CRes 0
            ans_level = f"ans_{CONFIDENCE_LEVELS[0]}"
            answer_prefix = random.choice(ANSWER_UNKNOWN_PREFIXES)
        elif correct_count <= 6:  # CRes 1-6
            ans_level = f"ans_{CONFIDENCE_LEVELS[1]}"
            subprefix = random.choice(ANSWER_UNCERTAIN_PREFIXES[correct_count])  # FIXME: format of PREFIXES changed
            answer_prefix = f"{ANSWER_UNCERTAIN_BASE_PREFIX} {subprefix}"
        else:  # CRes 7-8
            ans_level = f"ans_{CONFIDENCE_LEVELS[2]}"
            answer_prefix = random.choice(ANSWER_CONFIDENT_PREFIXES[correct_count])  # FIXME: format of PREFIXES changed

        data["visual_prefix"] = visual_prefix
        data["answer_prefix"] = answer_prefix
        data_by_conf_level[img_level][ans_level].append(data)

    # use randomly sampled examples
    balanced_data_list = []
    for img_level in img_conf_levels:
        for ans_level in ans_conf_levels:
            assert len(data_by_conf_level[img_level][ans_level]) >= n_each_area
            balanced_data_list.extend(
                random.sample(data_by_conf_level[img_level][ans_level], k=n_each_area)
            )

    assert len(balanced_data_list) == (len(CONFIDENCE_LEVELS) ** 2) * n_each_area
    random.shuffle(balanced_data_list)
    return balanced_data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, choices=["qwen", "llama", "llava"])
    parser.add_argument("--split-flag", type=str, choices=["t", "v"])
    parser.add_argument(
        "--n-each-area",
        type=int,
        help="number of examples to use in each confidence level area",
    )

    args = parser.parse_args()
    main(args)
