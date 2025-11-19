import argparse
import os

from method.base import BaseMethod
from method.fine_tuning import FineTuning
from method.utils import get_peft_dir
from common.const import DATASETS
from common.model_names import OpenSrcModel


def main(args):
    print("Running with the following arguments:")
    print(vars(args))

    kwargs = {
        "dataset_name": args.dataset_name,
        "model_name": args.model_key,
        "split_flag": args.split_flag,
        "response_sample_size": args.sample_size,
    }

    if args.use_peft_adapter:
        kwargs["peft_dir"] = get_peft_dir(args.dataset_name, args.model_key)
        method = FineTuning(**kwargs)
    else:
        method = BaseMethod(**kwargs)

    print(">>> Completed __init__")

    if not args.force and does_output_exist(args, method):
        print("Output file already exists. Aborting this run.")
        return

    method.generate_responses()
    print(">>> Completed generate_responses()")

    method.parse_structured_response()
    print(">>> Completed parse_structured_response()")


def does_output_exist(args, method_inst) -> bool:
    return os.path.exists(
        f"./output/{args.dataset_name}/response/{args.model_key}/{method_inst.output_name}.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-key", type=str, choices=[item.value for item in OpenSrcModel]
    )
    parser.add_argument("--dataset-name", type=str, choices=DATASETS)
    parser.add_argument("--split-flag", type=str, choices=["t", "v", "e"])
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--use-peft-adapter", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    main(args)
