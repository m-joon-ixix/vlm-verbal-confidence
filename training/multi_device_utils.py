import os
import torch
from trl import SFTTrainer

# environment variables
VISION_LAYERS_DEVICE_ENV = "FT_VISION_LAYERS_DEVICE"  # 0 recommended
LANGUAGE_LAYERS_DEVICE_ENV = "FT_LANGUAGE_LAYERS_DEVICE"  # 1 recommended


class CustomTwoDeviceSFTTrainerForVLM(SFTTrainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # overriding to move the input batch to the device in which submodule parameters reside
        inputs = self._move_to_submodule_device_for_parallel(inputs, model)
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        loss = loss.to(self.args.device)
        return (loss, outputs) if return_outputs else loss

    def _submodule_device(self, m):
        return next(m.parameters()).device

    def _move_to_submodule_device_for_parallel(self, inputs, model):
        # text pathway
        langauge_device = int(os.getenv(LANGUAGE_LAYERS_DEVICE_ENV))
        for k in ["input_ids", "attention_mask", "labels"]:
            if k in inputs and torch.is_tensor(inputs[k]):
                inputs[k] = inputs[k].to(
                    torch.device("cuda", langauge_device), non_blocking=True
                )

        # vision pathway
        vision_device = int(os.getenv(VISION_LAYERS_DEVICE_ENV))
        inputs["pixel_values"] = inputs["pixel_values"].to(
            torch.device("cuda", vision_device), non_blocking=True
        )

        return inputs


def get_mllama_device_map() -> dict:
    vision_device = int(os.getenv(VISION_LAYERS_DEVICE_ENV))
    language_device = int(os.getenv(LANGUAGE_LAYERS_DEVICE_ENV))

    # NOTE: The device map on a PEFT‑wrapped model can be inferred with the code below
    #   from peft import prepare_model_for_kbit_training, get_peft_model
    #   from accelerate import infer_auto_device_map
    #   base = prepare_model_for_kbit_training(base)  # base: pre-trained model
    #   model = get_peft_model(base, LoraConfig(...))
    #   infer_auto_device_map(model, max_memory={i: f"10GiB" for i in range(2)}, no_split_module_classes=["LlamaDecoderLayer"])
    device_map = {}

    for k in ["vision_model", "multi_modal_projector"]:
        device_map[f"model.{k}"] = vision_device

    for k in ["embed_tokens", "norm", "rotary_emb"]:
        device_map[f"model.language_model.{k}"] = language_device

    for i in range(38):
        device_map[f"model.language_model.layers.{i}"] = language_device

    for k in [
        "cross_attn_attn_gate",
        "cross_attn_mlp_gate",
        "cross_attn",
        "input_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp.act_fn",
        "post_attention_layernorm",
    ]:
        device_map[f"model.language_model.layers.38.{k}"] = language_device

    device_map["model.language_model.layers.39"] = language_device
    device_map["lm_head"] = language_device

    return device_map


def is_device_manually_assigned() -> bool:
    return (
        os.getenv(VISION_LAYERS_DEVICE_ENV) is not None
        and os.getenv(LANGUAGE_LAYERS_DEVICE_ENV) is not None
    )
