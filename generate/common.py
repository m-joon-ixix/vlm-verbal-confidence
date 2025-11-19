from common.model_names import (
    OPEN_SRC_MODEL_MAP,
    OpenSrcModel,
    model_name_to_dtype,
    DType,
)
from generate.unified import batch_query_unified
from generate.open_sourced import batch_query_open_sourced_llm


def batch_query_llm(
    prompt_list,
    model_name,
    peft_dir=None,
    max_new_tokens=1024,
    temperature=0.0,
    output_logits=False,
    batch_size=8,
):
    if model_name_to_dtype(model_name) != DType.OPEN_SRC:
        return batch_query_unified(
            prompt_list, model_name, max_new_tokens, temperature, output_logits
        )
    elif model_name in [k.value for k in OPEN_SRC_MODEL_MAP.keys()]:
        return batch_query_open_sourced_llm(
            prompt_list,
            OpenSrcModel(model_name),
            peft_dir,
            max_new_tokens,
            temperature,
            output_logits,
            batch_size
        )
    else:
        raise NotImplementedError
