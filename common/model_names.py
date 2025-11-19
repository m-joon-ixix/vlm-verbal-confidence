from enum import Enum


class DType(Enum):
    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"
    OPEN_SRC = "open_sourced"


def model_name_to_dtype(model_name: str) -> DType:
    if any([exp in model_name.lower() for exp in ["gpt", "o1", "o3", "o4"]]):
        return DType.OPENAI
    elif "claude" in model_name.lower():
        return DType.CLAUDE
    elif "gemini" in model_name.lower():
        return DType.GEMINI
    else:
        return DType.OPEN_SRC


class OpenSrcModel(Enum):
    LLAMA = "llama"
    LLAVA = "llava"
    QWEN = "qwen"
    INTERNVL = "internvl"


OPEN_SRC_MODEL_MAP = {
    OpenSrcModel.LLAMA: "meta-llama/Llama-3.2-11B-Vision-Instruct",
    OpenSrcModel.LLAVA: "llava-hf/llava-v1.6-mistral-7b-hf",
    OpenSrcModel.QWEN: "Qwen/Qwen2.5-VL-7B-Instruct",
    OpenSrcModel.INTERNVL: "OpenGVLab/InternVL3_5-8B-HF",
}

FLASH_ATTN_SUPPORTED_MODELS = [
    OpenSrcModel.QWEN,
    OpenSrcModel.LLAVA,
    OpenSrcModel.INTERNVL,
]
