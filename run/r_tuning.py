import argparse
from torch.cuda import is_available as is_cuda_available
from time import time
from datetime import timedelta

from method.r_tuning import RTuning
from method.r_tuning_peft import RTuningPeft
from method.r_tuning_peft_only_answer import RTuningPeftOnlyAnswer
from method.utils import get_peft_dir
from common.const import DATASETS
from common.model_names import OpenSrcModel

FUNCTIONS = [
    "generate_grounding",
    "generate_ground_truth",
    "extract_ground_truth",
    "generate_sample_image_desc",
    "generate_image_desc_confidence",
    "generate_answer_confidence",
    "determine_optimal_image_threshold",
    "determine_optimal_answer_threshold",
    "abstain_image_desc",  # 8
    "abstain_image_desc_confidence",
    "abstain_answer",
    "abstain_answer_confidence",
    "abstain_make_decisions",
]

# can only execute solely with the `function` argument
SOLELY_EXECUTABLE_FUNCTIONS = [
    "generate_responses",
    "parse_structured_response",
    "construct_dataset",
    "prepare_validation_data",
    "abstain",
]

CUDA_REQUIRED_FUNCTIONS = [
    "generate_responses",
    "generate_sample_image_desc",
    "abstain",
    "abstain_image_desc",
    "abstain_answer",
]


def main(args):
    print("Running with the following arguments:")
    print(vars(args))

    kwargs = {
        "dataset_name": args.dataset_name,
        "model_name": args.model_key,
        "split_flag": args.split_flag,
        "response_sample_size": args.response_sample_size,
        "threshold_quality_metric": args.threshold_quality_metric,
    }

    if args.peft:
        if args.trained_with_all_datasets:
            kwargs["peft_dir"] = get_peft_dir("all", args.model_key)
        else:
            kwargs["peft_dir"] = get_peft_dir(
                args.dataset_name,
                args.model_key,
                config_filename=args.peft_config_filename,
            )

        if args.peft_config_filename == "only_answer":
            method = RTuningPeftOnlyAnswer(**kwargs)
        else:
            method = RTuningPeft(**kwargs)
    else:
        method = RTuning(**kwargs)

    if args.function:
        assert args.function in (
            FUNCTIONS + SOLELY_EXECUTABLE_FUNCTIONS
        ) or args.function.startswith("abstain_")

        functions = [args.function]
        if args.function == "generate_responses":
            functions.append("parse_structured_response")  # automatically following.
    else:
        functions = FUNCTIONS[args.start_idx : (args.end_idx + 1)]

    for func in functions:
        if func in CUDA_REQUIRED_FUNCTIONS and not is_cuda_available():
            print(f"{func}() requires CUDA, but CUDA not available. Terminating")
            break

        run_function(method, func)


def run_function(method: RTuning, function_name: str):
    # dynamic way to call `method.generate_grounding()`
    function_to_call = getattr(method, function_name.replace("-", "_"))
    start_time = time()

    function_to_call()  # run

    time_formatted = str(timedelta(seconds=(time() - start_time)))
    print(f">>> Completed {function_name}() => Time Elapsed: {time_formatted}")
    print("=" * 100)
    print("\n" * 3)


# example run command
# CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true PYTHONPATH=. python run/r_tuning.py --model-key qwen --dataset-name MMMU --split-flag v
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-key", type=str, choices=[item.value for item in OpenSrcModel]
    )
    parser.add_argument("--dataset-name", type=str, choices=DATASETS)
    parser.add_argument("--split-flag", type=str, choices=["t", "v", "e"])
    parser.add_argument(
        "--function",
        type=str,
        required=False,
        help="if given, executes only that function",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        required=False,
        choices=list(range(len(FUNCTIONS))),
        default=0,
        help="if given, starts with the function on that idx (inclusive)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        required=False,
        choices=list(range(len(FUNCTIONS))),
        default=len(FUNCTIONS) - 1,
        help="if given, ends at the function on that idx (inclusive)",
    )
    parser.add_argument("--response-sample-size", type=int, default=8)
    parser.add_argument(
        "--threshold-quality-metric", type=str, default="abstain_accuracy"
    )
    parser.add_argument("--peft", action="store_true")
    parser.add_argument(
        "--peft-config-filename", type=str, required=False, default="default"
    )
    parser.add_argument(
        "--trained-with-all-datasets",
        action="store_true",
        help="use peft model trained with all datasets",
    )

    args = parser.parse_args()
    main(args)
