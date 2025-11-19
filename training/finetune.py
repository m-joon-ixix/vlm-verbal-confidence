import os
import torch
import argparse
import math
from time import time
from typing import List
from datetime import datetime, timedelta
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from peft.tuners.tuners_utils import (
    check_target_module_exists,
    _maybe_include_all_linear_layers,
)
from PIL import Image

from common.const import QWEN_IMAGE_MAX_PIXELS, DATASETS
from common.model_names import OpenSrcModel
from common.json_utils import load_from_json, dump_to_json
from common.yaml_utils import load_from_yaml, dump_to_yaml
from common.slack_utils import slack_notify
from generate.open_sourced import get_model, get_processor
from training.multi_device_utils import (
    CustomTwoDeviceSFTTrainerForVLM,
    get_mllama_device_map,
    is_device_manually_assigned,
)


PARAM_PREFIXES_TO_FREEZE = [
    # NOTE: these were listed from `for n, _ in model.named_parameters()` in qwen, llava, llama, internvl models
    "model.vision_tower",
    "model.multi_modal_projector",
    "model.vision_model",
    "model.visual",
    "model.image_newline",
]


def finetune(args):
    print("Running with the following arguments:")
    print(vars(args))

    if is_device_manually_assigned():
        import training.mllama_patch
        import training.torch_autograd_patch

    if _is_run_by_accelerate():
        # Assign each process launched by accelerate to a single CUDA device
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))

    model = get_model(args.model_key, device_map=_get_model_device_map(args))
    processor = get_processor(args.model_key)

    # Training arguments
    train_dataset = _get_finetune_dataset(args, "train")
    training_args = _get_training_args(
        args.model_key, args.dataset_name, len(train_dataset)
    )
    lora_config = _get_lora_config()
    lora_config.target_modules = _get_lora_target_modules(model)

    # Initialize Trainer
    trainer_cls = (
        CustomTwoDeviceSFTTrainerForVLM if is_device_manually_assigned() else SFTTrainer
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=_get_finetune_dataset(args, "val"),
        data_collator=_get_data_collator(args.model_key, processor),
        processing_class=processor,
        peft_config=lora_config,
    )

    # Train the model
    trainer.train()

    # Save model
    trainer.save_model(training_args.output_dir)

    # Manually save logs
    dump_to_json(
        f"{training_args.logging_dir}/log_history.json", trainer.state.log_history
    )

    # save training args in JSON format (by default, it only saves it into `.bin` format so it can only be read by `torch.load()`)
    save_training_args_to_json(training_args, lora_config)

    # Write down a description of this fine-tuning in the output_dir, if given
    if args.desc:
        _write_description(args.desc, training_args.output_dir)

    # # automatically write output dir into yaml config file
    # peft_dir_path = "./config/r_tuning/peft_dirs/default.yaml"
    # peft_dir_map = load_from_yaml(peft_dir_path)
    # peft_dir_map[args.dataset_name][args.model_key.value] = training_args.output_dir.split("/")[-1]
    # dump_to_yaml(peft_dir_path, peft_dir_map)

    slack_notify("Fine-tuning Completed!", output_dir=training_args.output_dir)


def _get_model_device_map(args):
    if args.model_key == OpenSrcModel.LLAMA and is_device_manually_assigned():
        return get_mllama_device_map()

    if _is_run_by_accelerate():
        print("Info: Accelerator usage was detected. using device_map=None")
        return None

    return "auto"


def _get_training_args(
    model_key: OpenSrcModel, dataset_name: str, training_set_size: int
) -> TrainingArguments:
    training_config = load_from_yaml("./config/training.yaml")

    # auto-compute `num_train_epochs`
    num_of_epochs = round(
        (training_config.pop("total_train_examples") + 1) / training_set_size
    )
    training_config["num_train_epochs"] = num_of_epochs
    print(f"Training Set Size: {training_set_size}, Training Epochs: {num_of_epochs}")

    current_time = datetime.now().strftime("%y%m%d-%H%M")
    output_dir = (
        f"./training_output/{dataset_name}/{model_key.value}/lora_{current_time}"
    )

    return SFTConfig(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        **training_config,
    )


def _get_lora_config() -> LoraConfig:
    return LoraConfig(**load_from_yaml("./config/lora.yaml"))


def _get_lora_target_modules(model) -> List[str]:
    # Freeze ViT & Projector params
    frozen_param_names = []
    for n, p in model.named_parameters():
        if any(n.startswith(prefix) for prefix in PARAM_PREFIXES_TO_FREEZE):
            p.requires_grad = False
            frozen_param_names.append(n)

    # Filter target modules to exclude frozen scopes
    lora_config = _get_lora_config()  # to use the initial `target_modules` setting
    # needed when target_modules="all-linear" => usually ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    lora_config = _maybe_include_all_linear_layers(lora_config, model)
    target_modules = []  # trainable modules
    for module_name, _ in model.named_modules():
        if not check_target_module_exists(lora_config, module_name):
            continue

        # skip if module contains any frozen parameter
        if any(param_name.startswith(module_name) for param_name in frozen_param_names):
            continue

        target_modules.append(module_name)

    return target_modules


def _get_finetune_dataset(args, split: str) -> Dataset:
    if args.method_name == "rtuning":
        filename = f"rtuning_base_{split}"
    elif args.method_name.startswith("balanced") and split != "train":  # val
        n = int(args.method_name.split("-")[-1])
        # divide the N in filename by 8
        filename = f"balanced_each-{int(n / 8)}_{split}"
    else:
        filename = f"{args.method_name}_{split}"

    dir_path = f"./output/{args.dataset_name}/training_data/{args.model_key.value}"
    if args.ft_dataset_subdir:
        dir_path += f"/{args.ft_dataset_subdir}"

    data_list = load_from_json(f"{dir_path}/{filename}.json")

    if is_device_manually_assigned():
        _reduce_if_too_many_images(data_list, split)

    return Dataset.from_list(data_list)


# Check each example in the training set, and reduce multiple image tokens in a single message into one image token.
#   NOTE: When training Llama-3.2-Vision, the vision layers are all put into a single device. (since error is raised when tensors are in different devices)
#         Thus, having more than 2 image tokens in the whole conversation text causes an OOM error on that device.
def _reduce_if_too_many_images(data_list: list, split: str):
    for example in data_list:
        img_cnt_was_reduced = False
        for message in example["messages"]:
            if message["role"] != "user":
                continue

            image_count = sum(
                content["type"] == "image" for content in message["content"]
            )
            if image_count <= 1:
                continue

            new_text = ""
            for content in message["content"]:
                if content["type"] == "text":
                    new_text += content["text"]
                elif content["type"] == "image":
                    new_text += "the image"

            message["content"] = [{"type": "image"}, {"type": "text", "text": new_text}]
            img_cnt_was_reduced = True

        if img_cnt_was_reduced:
            print(
                f"Images were reduced in the example with split: {split}, image_path: {example['image_path']}"
            )


def _get_data_collator(model_key: OpenSrcModel, processor):
    def collate_fn(dataset):
        """
        Args:
            dataset: List of dataset examples with 'messages' and 'images' fields

        Returns:
            Batch dictionary with input_ids, attention_mask, pixel_values, and labels
        """
        texts = []
        images = []
        for example in dataset:
            # Apply chat template to convert messages to text format
            _conversation_text = processor.apply_chat_template(
                example["messages"], tokenize=False
            )
            texts.append(_conversation_text)

            _image = Image.open(example["image_path"]).convert("RGB")
            w, h = _image.size
            # resize oversized images
            if w * h > QWEN_IMAGE_MAX_PIXELS:
                scale = math.sqrt(QWEN_IMAGE_MAX_PIXELS / float(w * h))
                _image = _image.resize(
                    # decrease scaled w, h to closest multiple of 28 (for visual tokenization grid alignment)
                    (int(w * scale / 28.0) * 28, int(h * scale / 28.0) * 28),
                    resample=Image.Resampling.BICUBIC,
                )

            # the number of images has to equal the number of image tokens in the string prompt
            image_count = _conversation_text.count(processor.image_token)
            images.append([_image] * image_count)

        # Process both text and images
        batch = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Create labels by cloning input_ids
        labels = batch["input_ids"].clone()

        # list of tokens to mask
        token_ids_to_mask = [
            # model-specific pad token ID
            processor.tokenizer.pad_token_id,
            # model-specific image token ID
            processor.tokenizer.convert_tokens_to_ids(processor.image_token),
        ]

        if model_key == OpenSrcModel.QWEN:
            # pre-defined vision-control tokens ("<|vision_start|>", "<|vision_end|>")
            # Reference: https://arxiv.org/pdf/2409.12191#page=6.51
            token_ids_to_mask.extend([151652, 151653])

        # Mask specific tokens in labels (set to -100 to ignore in loss computation)
        for token_id in set(token_ids_to_mask):
            labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def save_training_args_to_json(
    training_args: TrainingArguments, lora_config: LoraConfig
):
    lora_config_dict = lora_config.to_dict()
    dump_to_json(
        f"{training_args.output_dir}/lora_target_modules.json",
        lora_config_dict.pop("target_modules"),
    )

    training_args_dict = training_args.to_dict()
    training_args_dict["lora_config"] = lora_config_dict

    dump_to_json(f"{training_args.output_dir}/training_args.json", training_args_dict)


def _write_description(description: str, output_dir: str):
    output_path = f"{output_dir}/description.txt"
    with open(output_path, "w") as f:
        f.write(description)

    print(f"Wrote down description to: {output_path}")


def _is_run_by_accelerate() -> bool:
    return os.getenv("LOCAL_RANK") is not None


# NOTE: using accelerator:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --config_file config/accelerate_fsdp.yaml training/finetune.py --model-key llama --dataset-name MMMU --method-name rtuning --desc "2-Stage Training with Visual & Answer Prefixes"
# NOTE: to fine-tune Llama-3.2-Vision-11B by manually placing tensors on devices
#   CUDA_VISIBLE_DEVICES=0,1 FT_VISION_LAYERS_DEVICE=0 FT_LANGUAGE_LAYERS_DEVICE=1 PYTHONPATH=. python training/finetune.py --model-key llama --dataset-name MMMU --method-name rtuning --desc "2-Stage Training with Visual & Answer Prefixes"
# NOTE: to fine-tune using a training set where all datasets are mixed up
#   CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python training/finetune.py --model-key qwen --dataset-name all --method-name balanced_each-80 --desc "using balanced training set with 80 examples in each bin"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-key", type=OpenSrcModel)
    parser.add_argument("--dataset-name", type=str, choices=DATASETS + ["all"])
    parser.add_argument("--method-name", type=str)
    parser.add_argument(
        "--ft-dataset-subdir",
        type=str,
        required=False,
        help="Sub-directory to be appended after 'output/{dataset}/training_data/{model}'",
    )
    parser.add_argument("--desc", type=str, required=False)

    args = parser.parse_args()
    start_time = time()

    finetune(args)

    time_formatted = str(timedelta(seconds=(time() - start_time)))
    print(f"Training Completed (Time Elapsed: {time_formatted})")
