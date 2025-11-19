import os
import argparse

from method.utils import get_all_method_classes
from common.json_utils import load_from_json
from common.const import DATASETS


def main(args):
    print("Running with the following arguments:")
    print(vars(args))

    method_class = get_method_class(args.method_name)
    method = method_class(
        dataset_name=args.dataset_name,
        model_name=args.model_key,
        split_flag=args.split_flag,
    )
    print(">>> Completed __init__")

    if does_output_exist(args, method):
        print("Output file and abstain data already exists. Aborting this run.")
        return

    method.abstain()
    print(">>> Completed abstain()")


def get_method_class(method_name):
    result = [cls for cls in get_all_method_classes() if cls.method_name == method_name]
    if len(result) == 0:
        raise ValueError(f"Unsupported method_name: {method_name}")

    return result[0]


def does_output_exist(args, method_inst) -> bool:
    filepath = f"./output/{args.dataset_name}/response/{args.model_key}/{method_inst.output_name}.json"
    if not os.path.exists(filepath):
        return False

    # check if the "abstain" key exists in the data
    data = load_from_json(filepath)
    return "abstain" in data[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 'origin' 'noa' 'rtuning' 'cooperate' 'reflect' 'origin' 'analysis' 'verbalize' 'prob' 'consistency' 'origin'
    parser.add_argument("--method-name", type=str)
    parser.add_argument("--model-key", type=str, choices=["qwen", "llama", "llava"])
    parser.add_argument("--dataset-name", type=str, choices=DATASETS)
    parser.add_argument("--split-flag", type=str, choices=["t", "v", "e"])

    args = parser.parse_args()
    main(args)
