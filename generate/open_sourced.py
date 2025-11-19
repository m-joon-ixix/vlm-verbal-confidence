import os
import torch
import torch.nn.functional as F

from peft import PeftModel
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModel,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    MllamaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    InternVLForConditionalGeneration,
    BitsAndBytesConfig,
)

from common.const import MODEL_CACHE_DIR, QWEN_IMAGE_MAX_PIXELS
from common.yaml_utils import load_secret
from common.model_names import (
    OPEN_SRC_MODEL_MAP,
    OpenSrcModel,
    FLASH_ATTN_SUPPORTED_MODELS,
)

MODEL_CLASS_MAP = {
    OpenSrcModel.QWEN: Qwen2_5_VLForConditionalGeneration,
    OpenSrcModel.LLAMA: MllamaForConditionalGeneration,
    OpenSrcModel.LLAVA: LlavaNextForConditionalGeneration,
    OpenSrcModel.INTERNVL: InternVLForConditionalGeneration,
}


def batch_query_open_sourced_llm(
    prompt_list,
    model_key: OpenSrcModel,
    peft_dir=None,
    max_new_tokens=768,
    temperature=0.0,
    output_logits=False,
    batch_size=8,
):
    model = get_model(model_key, peft_dir)
    processor = get_processor(model_key)
    model.eval()

    generate_kwargs = _build_generate_kwargs(
        temperature, max_new_tokens, output_logits, processor
    )
    print(f"Generate KwArgs: {generate_kwargs}")

    responses = []
    log_probs = []
    knowledge_source_predictions = []

    print(f"An example of prompt (image part is not printed):")
    # check the `else` statement in the method `form_mm_query()`
    print(prompt_list[0][-1]["content"][0]["text"])

    print(f"[Generate] Model: {model_key.value}, Batch Size: {batch_size}")
    if peft_dir:
        print(f"Using adapters from peft_dir: {peft_dir}")

    with torch.no_grad():
        for i in tqdm(
            range(0, len(prompt_list), batch_size), desc=f"Generating Answers"
        ):
            begin = i
            end = min(i + batch_size, len(prompt_list))
            inputs = _build_inputs(prompt_list[begin:end], processor, model.device)

            outputs = model.generate(**inputs, **generate_kwargs)
            sequences = outputs.sequences[:, inputs["input_ids"].shape[-1] :].cpu()

            if output_logits:
                log_probs.extend(
                    _get_batch_log_probs(outputs, sequences, end - begin, processor)
                )

            if generate_kwargs["output_hidden_states"]:
                knowledge_source_predictions.extend(
                    _get_batch_knowledge_predictions(outputs, end - begin)
                )

            texts = processor.batch_decode(sequences, skip_special_tokens=True)
            responses += texts

            # clear memory after each batch to prevent memory usage from keep increasing
            del outputs, inputs
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return log_probs, responses


def get_model(model_key: OpenSrcModel, peft_dir=None, device_map="auto"):
    model_class = MODEL_CLASS_MAP.get(model_key, AutoModel)
    use_flash_attention = model_key in FLASH_ATTN_SUPPORTED_MODELS

    model_kwargs = _build_model_kwargs()
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = model_class.from_pretrained(
        OPEN_SRC_MODEL_MAP[model_key], **model_kwargs, device_map=device_map
    )

    if peft_dir is None:
        return model
    else:
        return PeftModel.from_pretrained(model, peft_dir)


def get_processor(model_key: OpenSrcModel):
    kwargs = {"token": load_secret("hf_key"), "trust_remote_code": True, "cache_dir": MODEL_CACHE_DIR}
    if model_key == OpenSrcModel.QWEN:
        kwargs["max_pixels"] = QWEN_IMAGE_MAX_PIXELS

    processor = AutoProcessor.from_pretrained(OPEN_SRC_MODEL_MAP[model_key], **kwargs)
    processor.tokenizer.padding_side = "left"

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return processor


def _build_model_kwargs() -> dict:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.bfloat16,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model_kwargs = {
        "token": load_secret("hf_key"),
        "trust_remote_code": True,
        "cache_dir": MODEL_CACHE_DIR,
        "torch_dtype": torch.bfloat16,
        "quantization_config": quantization_config,
    }

    return model_kwargs


def _build_generate_kwargs(
    temperature: float, max_new_tokens: int, output_logits: bool, processor
) -> dict:
    kwargs = {
        "do_sample": True if temperature != 0.0 else False,
        "top_p": 1.0,
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_logits": output_logits,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "output_hidden_states": output_logits,
    }

    # To prevent warning messages, set temperature only in sample-based generation
    if kwargs["do_sample"]:
        kwargs["temperature"] = temperature

    return kwargs


def _build_inputs(prompt_list, processor, device):
    text_inputs = processor.apply_chat_template(
        prompt_list,
        add_generation_prompt=True,
        tokenize=False,
    )

    image_inputs, _ = process_vision_info(prompt_list)
    if image_inputs:
        images = []  # List[List[Image]]
        image_idx = 0
        for text_input in text_inputs:
            image_count = text_input.count(processor.image_token)
            images.append(image_inputs[image_idx : (image_idx + image_count)])
            image_idx += image_count
    else:
        images = None

    inputs = processor(
        text=text_inputs, images=images, return_tensors="pt", padding=True
    ).to(device, dtype=torch.bfloat16)

    # logging (if enabled. OPEN_LLM_LOGGING=1, OPEN_LLM_LOGGING=0)
    if bool(int(os.getenv("OPEN_LLM_LOGGING", "0"))):
        with open("log.txt", "a") as f:
            f.write("inputs:" + str(inputs) + "\n")

    return inputs


def _get_batch_log_probs(outputs, sequences, batch_size, processor):
    logits = [logit.cpu() for logit in outputs.logits]
    log_prob = -F.log_softmax(torch.stack(logits, dim=1), dim=-1)
    answer_log_probs = log_prob.gather(-1, sequences[:, :, None]).squeeze(-1)

    log_probs = []
    for j in range(batch_size):
        valid_mask = (
            sequences[j, :] != processor.tokenizer.pad_token_id
        )  # delete pad token
        valid_log_prob = answer_log_probs[j, :][valid_mask]

        if len(valid_log_prob) > 0:
            log_probs.append(valid_log_prob.mean().item())
        else:
            log_probs.append(0.0)

    return log_probs


def _get_batch_knowledge_predictions(outputs, batch_size):
    hidden_states_list = [states.cpu() for states in outputs.hidden_states]

    # extract the last layer activations
    mlp_activations_list = [states[-1] for states in hidden_states_list]

    # Minjoon: no need to train the regression model???
    knowledge_classifier = LogisticRegression()
    knowledge_source_predictions = []
    for j in range(batch_size):
        sample_activation = mlp_activations_list[j].mean(dim=0).cpu().numpy()
        prediction = knowledge_classifier.predict([sample_activation])
        knowledge_source_predictions.append("PK" if prediction == 0 else "CK")

    return knowledge_source_predictions
