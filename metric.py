import numpy as np
import argparse
import os

from common.const import ABSTAIN_TOKEN, FAILED_TOKEN, DATASETS
from common.model_names import OpenSrcModel
from common.json_utils import load_from_json, dump_to_json
from common.yaml_utils import load_from_yaml
from method.utils import get_all_method_classes


def compute_metrics(
    ori_prediction_flags,
    ori_label_flags,
    ori_abstain_flags,
    abstain_scores=None,
    dtype="exact_match",
):
    """
    prediction_flags: a list of [predicted label, ABSTAIN_TOKEN, FAIL_TOKEN]s representing the prediction by the VLM
    label_flags: a list of [label] representing the labels of the dataset
    correct_flags: a list of [False, True]s representing the correctness of each QA answered by the VLM
    abstain_flags: a list of [False, True]s representing whether the LLM abstained from answering each QA
    abstain_scores: a list of floats from 0 to 1 representing the confidence of the VLM in abstaining
    returns: a dictionary of metrics
    """

    assert len(ori_prediction_flags) == len(ori_abstain_flags)
    assert len(ori_label_flags) == len(ori_abstain_flags)

    correct_flags = []
    abstain_flags = []

    if dtype == "exact_match":
        for pred, label, abstain in zip(
            ori_prediction_flags, ori_label_flags, ori_abstain_flags
        ):
            if (
                abstain != FAILED_TOKEN
                # and abstain != ABSTAIN_TOKEN
                # and pred != FAILED_TOKEN
                and pred != ABSTAIN_TOKEN
            ):
                correct_flags.append(pred == label)
                abstain_flags.append(abstain)
    else:
        raise NotImplementedError

    # group A: answered and correct
    # group B: abstained and correct
    # group C: answered and incorrect
    # group D: abstained and incorrect
    ans_corr = 0
    abs_corr = 0
    ans_incorr = 0
    abs_incorr = 0

    total_count = len(correct_flags)
    assert len(abstain_flags) == total_count

    for i in range(total_count):
        if abstain_flags[i]:
            if correct_flags[i]:
                abs_corr += 1
            else:
                abs_incorr += 1
        else:
            if correct_flags[i]:
                ans_corr += 1
            else:
                ans_incorr += 1

    assert ans_corr + abs_corr + ans_incorr + abs_incorr == total_count

    # accuracy: accuracy of the LLM on all of the questions
    accuracy = (ans_corr + abs_corr) / total_count

    # reliable accuracy: accuracy of the LLM on the questions it answered
    try:
        reliable_accuracy = ans_corr / (ans_corr + ans_incorr)
    except ZeroDivisionError:
        reliable_accuracy = None

    # effective reliability: correct 1, incorrect -1, abstained 0
    effective_reliability = (ans_corr - ans_incorr) / total_count

    # abstain accuracy: accuracy of the LLM abstain decisions, how many times correct_flags == !abstain flags
    abstain_accuracy = (ans_corr + abs_incorr) / total_count

    # abstain precision: how many abstains is right among all abstains
    try:
        abstain_precision = abs_incorr / (abs_corr + abs_incorr)
    except ZeroDivisionError:
        abstain_precision = None

    # abstain recall: how many abstains is right among all incorrect answers
    try:
        abstain_recall = abs_incorr / (ans_incorr + abs_incorr)
    except ZeroDivisionError:
        abstain_recall = None

    try:
        abstain_f1 = (2 * abstain_precision * abstain_recall) / (
            abstain_precision + abstain_recall
        )
    except (TypeError, ZeroDivisionError):
        abstain_f1 = None

    # abstain ECE: bucket abstain confidence into 10 buckets (0:0.1:1), compute the expected calibration error
    if abstain_scores is not None and max(abstain_scores) != min(abstain_scores):
        bucket_ece = compute_bucket_ece(abstain_scores, correct_flags)
    else:
        bucket_ece = None

    # abstain rate: what percentage of questions the LLM abstained from
    abstain_rate = (abs_corr + abs_incorr) / total_count

    return {
        "accuracy": accuracy,
        "reliable_accuracy": reliable_accuracy,
        "effective_reliability": effective_reliability,
        "abstain_accuracy": abstain_accuracy,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "abstain_f1": abstain_f1,
        "abstain_ece": bucket_ece,
        "abstain_rate": abstain_rate,
    }


def compute_bucket_ece(abstain_scores, correct_flags) -> float:
    # rescale abstain scores to 0-1 before calculation
    max_score = max(abstain_scores)
    min_score = min(abstain_scores)
    for i in range(len(abstain_scores)):
        abstain_scores[i] = (abstain_scores[i] - min_score) / (max_score - min_score)

    bucket_probs = [[] for i in range(10)]
    bucket_abstain = [[] for i in range(10)]  # whether it should have abstained

    for i in range(len(abstain_scores)):
        if abstain_scores[i] == 1:
            bucket = 9
        else:
            bucket = int(abstain_scores[i] * 10)
        bucket_probs[bucket].append(abstain_scores[i])
        if correct_flags[i] == 1:
            bucket_abstain[bucket].append(0)
        else:
            bucket_abstain[bucket].append(1)

    bucket_ece = 0
    for i in range(10):
        if len(bucket_probs[i]) == 0:
            continue
        bucket_probs_avg = np.mean(bucket_probs[i])
        bucket_abstain_avg = np.mean(bucket_abstain[i])
        bucket_ece += abs(bucket_abstain_avg - bucket_probs_avg) * len(bucket_probs[i])

    bucket_ece /= len(abstain_scores)
    return bucket_ece


def compute_metrics_from_dataset(data_list):
    prediction_list = []
    label_list = []
    abstain_list = []
    for data in data_list:
        if "abstain_response" in data:
            prediction_list.append(data["extracted_abstain_response"])
            # NOTE: "abstain_response" is generated on the first of "sampled_idxs"
            label_list.append(data["sampled_idxs"][0].index(data["correct_answer_idx"]))
            assert len(data["abstain"]) == 1
            abstain_list += data["abstain"]
        else:
            response_sample_size = len(data["extracted_responses"])
            assert len(data["sampled_idxs"]) == response_sample_size

            prediction_list += data["extracted_responses"]
            label_list += [
                idx.index(data["correct_answer_idx"]) for idx in data["sampled_idxs"]
            ]

            if "abstain" not in data:
                # if there is no "abstain" field, consider that they all do not abstain
                abstain_list += [False] * response_sample_size
            elif len(data["abstain"]) == response_sample_size:
                abstain_list += data["abstain"]
            elif len(data["abstain"]) == 1:
                abstain_list += data["abstain"] * response_sample_size
            else:
                raise AssertionError("Unexpected length of data['abstain']")

    return compute_metrics(prediction_list, label_list, abstain_list)


def compute_and_save_metrics(dataset_name: str, split: str, args):
    stat_path = get_stats_filepath(args.method, dataset_name, split, args.peft)
    if os.path.exists(stat_path):
        stats = load_from_json(stat_path)  # overwrite on existing stats
    else:
        stats = {}

    for model in OpenSrcModel:
        resp_path = get_response_filepath(dataset_name, model.value, split, args)
        if resp_path is None or not os.path.exists(resp_path):
            continue

        model_key = get_stats_dict_key(dataset_name, model.value, args)
        data_list = load_from_json(resp_path)
        stats[model_key] = compute_metrics_from_dataset(data_list)

    dump_to_json(stat_path, stats)


def get_stats_filepath(method: str, dataset_name: str, split: str, peft: bool) -> str:
    if method == "rtuning":
        stats_filename = f"rtuning_peft_{split}" if peft else f"rtuning_base_{split}"
    else:
        stats_filename = f"{method}_{split}"

    return f"./output/{dataset_name}/stats/{stats_filename}.json"


def get_stats_dict_key(dataset_name: str, model_name: str, args) -> str:
    if args.peft:
        peft_dir_map = load_from_yaml(
            f"./config/r_tuning/peft_dirs/{args.peft_config_filename}.yaml",
            print_msg=False,
        )
        peft_subdir = peft_dir_map[
            "all" if args.trained_with_all_datasets else dataset_name
        ][model_name]
        return f"{model_name}_{peft_subdir}"
    else:
        return model_name


def get_response_filepath(dataset_name: str, model_name: str, split: str, args) -> str:
    if args.peft:
        peft_dir_map = load_from_yaml(
            f"./config/r_tuning/peft_dirs/{args.peft_config_filename}.yaml",
            print_msg=False,
        )
        peft_subdir = peft_dir_map.get(
            "all" if args.trained_with_all_datasets else dataset_name, {}
        ).get(model_name)

        if peft_subdir is None:
            return None

        resp_filename = f"{args.method}_{peft_subdir}_{split}"
    else:
        resp_filename = (
            f"rtuning_base_{split}"
            if args.method == "rtuning"
            else f"{args.method}_{split}"
        )

    return f"./output/{dataset_name}/response/{model_name}/{resp_filename}.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        required=False,
        help="If not given, runs on all splits",
    )
    parser.add_argument("--peft", action="store_true")
    parser.add_argument(
        "--peft-config-filename", type=str, required=False, default="default"
    )
    parser.add_argument(
        "--trained-with-all-datasets",
        action="store_true",
        help="export stats about results from the model fine-tuned with all datasets",
    )
    args = parser.parse_args()

    assert args.method in [_cls.method_name for _cls in get_all_method_classes()]

    if args.peft:
        assert args.method in ["origin", "rtuning"]

    if args.method == "ft":
        assert args.peft

    if args.split:
        splits = [args.split]
    else:
        splits = ["train", "val", "test"]

    for dataset_name in DATASETS:
        for split in splits:
            compute_and_save_metrics(dataset_name, split, args)
