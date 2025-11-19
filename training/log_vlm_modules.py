import os
import argparse
from peft import get_peft_model

from common.model_names import OPEN_SRC_MODEL_MAP
from generate.open_sourced import get_model
from training.finetune import _get_lora_target_modules, _get_lora_config


def main(args):
    for model_key in OPEN_SRC_MODEL_MAP.keys():
        view_modules(model_key, args)


def view_modules(model_key, args):
    model = get_model(model_key)
    if args.lora:
        lora_config = _get_lora_config()
        lora_config.target_modules = _get_lora_target_modules(model)
        write_to_txt(
            f"./output/vlm_modules/{model_key.value}/lora_target_modules.txt",
            lora_config.target_modules,
        )

        model = get_peft_model(model, peft_config=lora_config)

    parameter_names = [n for n, _ in model.named_parameters()]
    module_names = [n for n, _ in model.named_modules()]

    write_to_txt(
        f"./output/vlm_modules/{model_key.value}/parameter_names{'_withLora' if args.lora else ''}.txt",
        parameter_names,
    )

    write_to_txt(
        f"./output/vlm_modules/{model_key.value}/module_names{'_withLora' if args.lora else ''}.txt",
        module_names,
    )


def write_to_txt(filepath: str, l: list):
    dir_name = os.path.dirname(filepath)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(filepath, "w") as f:
        f.write("\n".join(l))

    print(f"File written to: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()
    main(args)
