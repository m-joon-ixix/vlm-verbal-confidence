import re
import os
import importlib
import inspect
import numpy as np
from tqdm import tqdm
from typing import List, Optional

from common.const import (
    ABSTAIN_TOKEN,
    FAILED_TOKEN,
    IDX_TO_LETTER,
    LETTER_TO_IDX,
    PROHIBITED_CONTENT_MESSAGE,
)
from common.slack_utils import slack_notify
from common.yaml_utils import load_from_yaml


def clean_extracted_answers(
    dataset: list,
    pattern,
    post_process=lambda x: x,
    key="extracted_responses",
    new_key=None,
):
    if new_key is None:
        new_key = key
    pattern = re.compile(pattern)
    for data in dataset:
        new_extracted_answers = []
        for d in data[key]:
            match = pattern.search(d)
            if match:
                result = match.group(1)
                result = post_process(result)
            elif "none" in d.lower():
                result = ABSTAIN_TOKEN
            else:
                result = FAILED_TOKEN
            new_extracted_answers.append(result)
        data[new_key] = new_extracted_answers


def extract_multichoice_answers(data_list: list, response_attr: str = "responses"):
    for data in data_list:
        # pattern differs by the number of options. If there are 4 options, it looks for A-D
        last_letter = IDX_TO_LETTER[len(data["options"]) - 1]
        # match a single uppercase letter that is not preceded or followed by any letter/digit
        pattern = re.compile(rf"(?<![A-Za-z0-9])([A-{last_letter}])(?![A-Za-z0-9])")

        if isinstance(data[response_attr], list):
            data[f"extracted_{response_attr}"] = [
                _extract_multichoice_answer(
                    response,
                    pattern,
                    [data["options"][idx] for idx in data["sampled_idxs"][sample_num]],
                )
                for sample_num, response in enumerate(data[response_attr])
            ]
        elif isinstance(data[response_attr], str):
            data[f"extracted_{response_attr}"] = _extract_multichoice_answer(
                data[response_attr],
                pattern,
                [data["options"][idx] for idx in data["sampled_idxs"][0]],
            )
        else:
            raise TypeError(f"'{response_attr}' should be a list or str")


def _extract_multichoice_answer(response: str, pattern, ordered_options: List[str]):
    # parsing for a single response
    lines = response.strip().split("\n")
    lines = [line.strip() for line in lines]  # strip each line
    lines = [line for line in lines if len(line) > 0]  # remove empty lines

    # first, look for the upper case letter representing an option
    for idx in [-1, -2, 0, 1, -3]:
        try:
            match = pattern.search(lines[idx])
            if match:
                return LETTER_TO_IDX[match.group(1)]
        except (IndexError, KeyError):
            continue

    # if letter was not matched, look for the raw answer text
    for line_idx in [-1, 0]:
        matched_option_idxs = []
        for option_idx, option in enumerate(ordered_options):
            if option.lower() in lines[line_idx].lower():
                matched_option_idxs.append(option_idx)

        # If only one option was matched, return it. If more than one was matched we are unsure which is the answer.
        if len(matched_option_idxs) == 1:
            return matched_option_idxs[0]

    # if the code reached here, it means that it failed to match
    if "none" in response.lower():
        return ABSTAIN_TOKEN
    else:
        return FAILED_TOKEN


def extract_score_from_response(response: str, prefix: str):
    text = response.split(prefix)[-1].strip()
    if len(text) == 0:
        return FAILED_TOKEN

    if text[0] == "1":
        if len(text) > 1 and text[1] == "0":
            return 10
        else:
            return 1
    else:
        try:
            return int(text[0])
        except ValueError:
            return FAILED_TOKEN


def find_optimal_abstention_threshold(
    pred_flags,
    label_flags,
    ori_prob_flags,
    quality_metric="abstain_accuracy",
    tolerance=None,
) -> float:
    assert len(pred_flags) == len(label_flags)
    assert len(pred_flags) == len(ori_prob_flags)

    correct_flags = []
    prob_flags = []
    for pred, label, prob in zip(pred_flags, label_flags, ori_prob_flags):
        if pred != ABSTAIN_TOKEN and pred != FAILED_TOKEN and prob != FAILED_TOKEN:
            prob_flags.append(prob)
            correct_flags.append(pred == label)

    print(f"Quality metric used to find optimal threshold: {quality_metric}")
    if quality_metric == "abstain_accuracy":
        return find_threshold_max_abstain_accuracy(correct_flags, prob_flags, tolerance)
    elif quality_metric == "effective_reliability":
        return find_threshold_max_effective_reliability(
            correct_flags, prob_flags, tolerance
        )
    else:
        raise NotImplementedError


def find_threshold_max_abstain_accuracy(
    correct_flags: List[bool], prob_flags: List[float], tolerance: Optional[float]
) -> float:
    # "maximizing abstain_accuracy" <=> "minimizing wrong abstention decisions (errors)"
    min_error = float("inf")
    best_thresholds = []
    for threshold in tqdm(range(1, 100), desc="Looking for best threshold"):
        error = 0
        abstain = 0
        for i in range(len(correct_flags)):
            if prob_flags[i] < float(threshold / 100.0):
                if correct_flags[i]:
                    error += 1
                abstain += 1
            else:
                if not correct_flags[i]:
                    error += 1

        if tolerance is None or abstain / len(correct_flags) <= tolerance:
            if error < min_error:
                min_error = error
                best_thresholds = [float(threshold / 100.0)]
            elif error == min_error:
                # if error is tied, record all the thresholds
                best_thresholds.append(float(threshold / 100.0))

    # use the median of best thresholds as the final threshold
    return float(np.median(best_thresholds))


def find_threshold_max_effective_reliability(
    correct_flags: List[bool], prob_flags: List[float], tolerance: Optional[float]
) -> float:
    total_count = len(correct_flags)

    # NOTE: effective_reliability = (ans_corr - ans_incorr) / total_count
    #   choose the threshold that yields the maximum numerator value
    max_er_numerator = float("-inf")  # max effective_reliability numerator
    best_thresholds = []
    for threshold in tqdm(range(1, 100), desc="Looking for best threshold"):
        ans_corr = 0
        ans_incorr = 0
        abstain = 0
        for i in range(total_count):
            if prob_flags[i] < float(threshold / 100.0):
                abstain += 1
            else:
                if correct_flags[i]:
                    ans_corr += 1
                else:
                    ans_incorr += 1

        _er_numerator = ans_corr - ans_incorr
        if tolerance is None or abstain / total_count <= tolerance:
            if _er_numerator > max_er_numerator:
                max_er_numerator = _er_numerator
                best_thresholds = [float(threshold / 100.0)]
            elif _er_numerator == max_er_numerator:
                # if quality metric is tied, record all the thresholds
                best_thresholds.append(float(threshold / 100.0))

    # use the median of best thresholds as the final threshold
    return float(np.median(best_thresholds))


def count_correct_answers(data) -> tuple[int, float]:
    """
    Return:
        A tuple consisting of the number of correct answers and the ratio of correct answers
    """
    num_of_responses = len(data["extracted_responses"])

    correct_count = 0
    for i in range(num_of_responses):
        if (
            data["extracted_responses"][i] == ABSTAIN_TOKEN
            or data["extracted_responses"][i] == FAILED_TOKEN
        ):
            continue

        if data["extracted_responses"][i] == data["sampled_idxs"][i].index(
            data["correct_answer_idx"]
        ):
            correct_count += 1

    correct_ratio = correct_count / num_of_responses if num_of_responses > 0 else 0.0
    return correct_count, correct_ratio


def get_all_method_classes():
    classes = []
    for filename in os.listdir("./method"):
        if filename.endswith(".py") and filename != "__init__.py":
            module = importlib.import_module(f"method.{filename[:-3]}")
            for _, obj in inspect.getmembers(module, inspect.isclass):
                # only classes defined in this module
                if obj.__module__ == module.__name__:
                    classes.append(obj)

    return classes


def get_peft_dir(
    dataset_name: str,
    model_key: str,
    print_logs: bool = True,
    config_filename: str = "default",
) -> str:
    config_path = f"./config/r_tuning/peft_dirs/{config_filename}.yaml"

    peft_dir_config = load_from_yaml(config_path, print_msg=print_logs)
    peft_dir = f"./training_output/{dataset_name}/{model_key}/{peft_dir_config[dataset_name][model_key]}"

    assert os.path.exists(peft_dir)
    if print_logs:
        print(f"Using peft_dir: {peft_dir}")

    return peft_dir


def check_generate_failure(
    responses: List[str], attribute: str, model_name: str, output_path: str
):
    failed_indices = []
    prohibited_content_indices = []
    for idx, response in enumerate(responses):
        if response == FAILED_TOKEN:
            failed_indices.append(idx)

        if response == PROHIBITED_CONTENT_MESSAGE:
            prohibited_content_indices.append(idx)

    notify_params = {
        "output_path": output_path.split("output")[-1].replace(".json", "")[1:],
        "attribute": attribute,
        "model_name": model_name,
    }

    if len(failed_indices) > 0:
        slack_notify(
            f"Failed to generate for examples at these indices: {failed_indices}",
            **notify_params,
        )

    if len(prohibited_content_indices) > 0:
        slack_notify(
            f"Generate request was blocked due to PROHIBITED_CONTENT for examples at these indices: {prohibited_content_indices}",
            **notify_params,
        )
