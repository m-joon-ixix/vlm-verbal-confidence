import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common.const import MODEL_CACHE_DIR
from common.yaml_utils import load_secret

# https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def get_nli_model():
    return AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME,
        token=load_secret("hf_key"),
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
    ).to("cuda")


def get_nli_tokenizer():
    return AutoTokenizer.from_pretrained(NLI_MODEL_NAME)


def get_nli_probs(premise, hypothesis, model, tokenizer):
    """
    Returns:
        list of probabilities for each class: ["entailment", "neutral", "contradiction"]
    """
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(torch.device("cuda")))
    return torch.softmax(output["logits"][0], -1).tolist()
