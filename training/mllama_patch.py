import os
import torch
import transformers.models.mllama.modeling_mllama as tmmm

from training.multi_device_utils import LANGUAGE_LAYERS_DEVICE_ENV, is_device_manually_assigned

orig_prepare_cross_attention_mask = tmmm._prepare_cross_attention_mask


def custom_prepare_cross_attention_mask(cross_attention_mask, num_vision_tokens, dtype):
    cross_attention_mask, full_text_row_masked_out_mask = (
        orig_prepare_cross_attention_mask(
            cross_attention_mask, num_vision_tokens, dtype
        )
    )
    if is_device_manually_assigned():
        full_text_row_masked_out_mask = full_text_row_masked_out_mask.to(
            torch.device("cuda", int(os.getenv(LANGUAGE_LAYERS_DEVICE_ENV)))
        )

    return cross_attention_mask, full_text_row_masked_out_mask


tmmm._prepare_cross_attention_mask = custom_prepare_cross_attention_mask
