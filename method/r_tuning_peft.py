import os
import numpy as np
import torch
from tqdm import tqdm

from common.const import FAILED_TOKEN
from common.model_names import model_name_to_dtype
from generate.common import batch_query_llm
from generate.form_query import form_multi_turn_queries_for_finetuned_model
from generate.nli import get_nli_model, get_nli_tokenizer, get_nli_probs
from method.r_tuning_prefixes import *
from method.r_tuning import RTuning
from method.utils import extract_score_from_response


# The class to use when running with 2-stage (visual & answer) finetuned models
class RTuningPeft(RTuning):
    def __init__(self, **kwargs):
        assert "peft_dir" in kwargs
        kwargs["feedback_name"] = kwargs["peft_dir"].split("/")[-1]
        super().__init__(**kwargs)

        # if data was loaded from `origin` output file, remove `responses` attributes from the data loaded onto memory
        if not kwargs.get("load_from_exist", True) or not os.path.exists(
            f"./output/{self.dataset_name}/response/{self.model_name}/{self.output_name}.json"
        ):
            for i, data in enumerate(self.dataset.get_split(self.split_flag)):
                self.dataset.get_split(self.split_flag)[i] = {
                    k: v
                    for k, v in data.items()
                    if k not in ["responses", "extracted_responses"]
                }

    def generate_responses(self):
        # NOTE: the prompt needs to be a 3-turn conversation between user & assistant
        #   user querying for the image desc => assistant answering the image desc => user querying for the answer
        if not all(
            "greedy_image_desc" in data
            for data in self.dataset.get_split(self.split_flag)
        ):
            print(
                "[INFO] A greedy image desc is needed to generate responses. Pre-generating image descriptions with zero temperature."
            )
            self.generate_greedy_image_desc(target_attr="greedy_image_desc")

        query_list = form_multi_turn_queries_for_finetuned_model(
            self.dataset.get_split(self.split_flag),
            dtype=model_name_to_dtype(self.model_name),
            sample_size=self.sample_size,
            image_desc_attr="greedy_image_desc",
            target_attr="responses",
        )

        _, responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            peft_dir=self.peft_dir,
            temperature=0.0,
            batch_size=self.sample_size,
        )

        for i, data in enumerate(self.dataset.get_split(self.split_flag)):
            data["responses"] = responses[
                i * self.sample_size : (i + 1) * self.sample_size
            ]

        self.save_response()

    def prepare_validation_data(self):
        self.generate_sample_image_desc()
        self.generate_image_desc_confidence()
        self.generate_answer_confidence()
        self.determine_optimal_image_threshold()
        self.determine_optimal_answer_threshold()

    def generate_sample_image_desc(self):
        if all(
            "greedy_image_desc" in data
            for data in self.dataset.get_split(self.split_flag)
        ):
            print("[INFO] Greedy Image Desc already exists. Using the existing ones.")
            for data in self.dataset.get_split(self.split_flag):
                data["sample_image_desc"] = [data["greedy_image_desc"]]

            self.save_response()
        else:
            # in order to use confidence of image descriptions generated with temp=0, when determining threshold
            super().generate_sample_image_desc(sample_size=1, temperature=0.0)

    # normally exectued on validation set, before computing the abstention threshold
    def generate_image_desc_confidence(self):
        nli_model = get_nli_model()
        nli_tokenizer = get_nli_tokenizer()
        for data in tqdm(
            self.dataset.get_split(self.split_flag),
            desc="confidence of sampled image desc (NLI)",
        ):
            scores = [
                get_confidence_nli(
                    image_desc, SCORE_TO_VISUAL_PREFIXES, nli_model, nli_tokenizer
                )
                for image_desc in data["sample_image_desc"]
            ]

            data["image_desc_confidence_scores"] = scores
            data["image_desc_confidence"] = float(np.average(scores))

        self.save_response()

    # normally exectued on validation set, before computing the abstention threshold
    def generate_answer_confidence(self):
        nli_model = get_nli_model()
        nli_tokenizer = get_nli_tokenizer()
        for data in tqdm(
            self.dataset.get_split(self.split_flag),
            desc="confidence of generated responses (NLI)",
        ):
            scores = [
                get_confidence_nli(
                    response, SCORE_TO_ANSWER_PREFIXES, nli_model, nli_tokenizer
                )
                for response in data["responses"]
            ]

            data["answer_confidence_scores"] = scores
            data["answer_confidence"] = float(np.average(scores))

        self.save_response()

    def _prob_flags_for_answer_threshold_calculation(self, data):
        return [score / 10.0 for score in data["answer_confidence_scores"]]

    # measure the confidence of image desc by requerying the fine-tuned model
    def abstain_image_desc_confidence_requery(self):
        with open(
            "instruction/abstain_reflect_image_desc_score.txt", encoding="utf-8"
        ) as f:
            description_score_instruction = "".join(f.readlines())

        query_list = self.dataset.customize_queries(
            key="abstain_image_desc_confidence",
            dtype=model_name_to_dtype(self.model_name),
            split_flag=self.split_flag,
            sample_size=self.sample_size,
            instruction_func=lambda data, i: description_score_instruction.format(
                data["abstain_image_desc"]
            ),
            include_image=False,
            include_option=False,
        )

        _, score_responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            peft_dir=self.peft_dir,
            temperature=0.5,
            batch_size=self.sample_size,
        )

        for idx, data in enumerate(self.dataset.get_split(self.split_flag)):
            _scores = [
                extract_score_from_response(score_response, "Score: ")
                for score_response in score_responses[
                    idx * self.sample_size : (idx + 1) * self.sample_size
                ]
            ]
            _scores = [_score for _score in _scores if _score != FAILED_TOKEN]

            data["abstain_image_desc_confidence"] = (
                np.mean(_scores) if len(_scores) > 0 else FAILED_TOKEN
            )

        self.save_response()

    def abstain_image_desc_confidence(self):
        nli_model = get_nli_model()
        nli_tokenizer = get_nli_tokenizer()
        for data in tqdm(
            self.dataset.get_split(self.split_flag), desc="image desc confidence (NLI)"
        ):
            data["abstain_image_desc_confidence"] = get_confidence_nli(
                data["abstain_image_desc"],
                SCORE_TO_VISUAL_PREFIXES,
                nli_model,
                nli_tokenizer,
            )

        self.save_response()

    def abstain_answer(self):
        query_list = form_multi_turn_queries_for_finetuned_model(
            self.dataset.get_split(self.split_flag),
            dtype=model_name_to_dtype(self.model_name),
            sample_size=1,
            image_desc_attr="abstain_image_desc",
            target_attr="abstain_response",
        )

        _, responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            peft_dir=self.peft_dir,
            temperature=0.0,
            batch_size=1,
            # output_logits=True
        )

        for i, data in enumerate(self.dataset.get_split(self.split_flag)):
            data["abstain_response"] = responses[i]

        self.parse_structured_response(response_attr="abstain_response")
        # `self.save_response()` included above

    # measure the confidence of answer by requerying the fine-tuned model
    def abstain_answer_confidence_requery(self):
        with open("instruction/abstain_requery_instruction.txt", encoding="utf-8") as f:
            requery_instruction = "".join(f.readlines())

        requery_list = self.dataset.customize_queries(
            key="abstain_answer_confidence",
            dtype=model_name_to_dtype(self.model_name),
            split_flag=self.split_flag,
            sample_size=self.sample_size,
            instruction_func=lambda data, i: requery_instruction.format(
                response=data["abstain_response"]
            ),
            include_question=False,
            include_image=False,
            include_option=False,
        )

        _, requery_responses = batch_query_llm(
            requery_list,
            model_name=self.model_name,
            peft_dir=self.peft_dir,
            temperature=0.5,
            batch_size=self.sample_size,
        )

        for i, data in enumerate(self.dataset.get_split(self.split_flag)):
            _scores = [
                extract_score_from_response(requery_response, "Confidence: ")
                for requery_response in requery_responses[
                    i * self.sample_size : (i + 1) * self.sample_size
                ]
            ]
            _scores = [_score for _score in _scores if _score != FAILED_TOKEN]

            data["abstain_answer_confidence"] = (
                np.mean(_scores) if len(_scores) > 0 else FAILED_TOKEN
            )

        self.save_response()

    def abstain_answer_confidence(self):
        nli_model = get_nli_model()
        nli_tokenizer = get_nli_tokenizer()
        for data in tqdm(
            self.dataset.get_split(self.split_flag), desc="answer confidence (NLI)"
        ):
            data["abstain_answer_confidence"] = get_confidence_nli(
                data["abstain_response"],
                SCORE_TO_ANSWER_PREFIXES,
                nli_model,
                nli_tokenizer,
            )

        self.save_response()


# NOTE: score is in range [0, 10]
SCORE_TO_VISUAL_PREFIXES = {
    0: VISUAL_UNKNOWN_PREFIXES,
    5: VISUAL_UNCERTAIN_PREFIXES,
    10: VISUAL_CONFIDENT_PREFIXES,
}

# NOTE: score is in range [0, 10]
SCORE_TO_ANSWER_PREFIXES = {
    0: ANSWER_UNKNOWN_PREFIXES,
    5: ANSWER_UNCERTAIN_PREFIXES,
    10: ANSWER_CONFIDENT_PREFIXES,
}


def get_confidence_nli(
    response: str, score_to_prefixes: dict, model, tokenizer
) -> float:
    """
    Args:
        response: image_desc or answer
        score_to_prefixes: SCORE_TO_VISUAL_PREFIXES or SCORE_TO_ANSWER_PREFIXES
    """
    grid_scores = []
    maxpooled_entailments = []
    for score, prefixes in score_to_prefixes.items():
        grid_scores.append(score)

        _entailment_diffs = []
        for prefix in prefixes:
            # NOTE: check the entailment of the first n sentences (n: number of sentences in prefix)
            _n = len(prefix.split(".")) - 1
            head_response = ".".join(response.split(".")[0:_n]) + "."
            _probs = get_nli_probs(head_response, prefix, model, tokenizer)

            # diff of entailment, contradiction probabilities
            _entailment_diffs.append(_probs[0] - _probs[2])

        maxpooled_entailments.append(max(_entailment_diffs))  # max-pooling

    weights = torch.softmax(
        # multiply by 3 before applying softmax, to sharpen the distribution of weights
        torch.tensor(maxpooled_entailments, dtype=torch.float32) * 3,
        dim=0,
    ).tolist()

    final_score = 0.0
    for weight, grid_score in zip(weights, grid_scores):
        final_score += weight * grid_score

    return final_score
