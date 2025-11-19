import os
import numpy as np
import random
from scipy.stats import entropy
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

from method.base import BaseMethod
from method.utils import (
    find_optimal_abstention_threshold,
    count_correct_answers,
    check_generate_failure,
)
from generate.common import batch_query_llm
from generate.form_query import (
    build_finetune_user_content,
    build_finetune_assistant_content,
)
from common.const import (
    IDX_TO_LETTER,
    FAILED_TOKEN,
    QUALITY_METRIC_TO_POSTFIX,
    CONFIDENCE_LEVELS,
)
from common.model_names import model_name_to_dtype
from common.json_utils import load_from_json, dump_to_json
from common.random_utils import get_seed
from method.r_tuning_prefixes import (
    VISUAL_UNKNOWN_PREFIXES,
    VISUAL_UNCERTAIN_PREFIXES,
    VISUAL_CONFIDENT_PREFIXES,
    ANSWER_UNKNOWN_PREFIXES,
    ANSWER_UNCERTAIN_PREFIXES,
    ANSWER_CONFIDENT_PREFIXES,
)

JURY_MODEL_KEY_TO_NAME = {
    "gpt4o": "gpt-4o-mini-2024-07-18",
    "o4": "o4-mini-2025-04-16",
    "gemini": "gemini-2.5-flash-lite",
}

JUDGE_MODEL_NAME = "chatgpt-4o-latest"


class RTuning(BaseMethod):
    method_name = "rtuning"

    def __init__(
        self,
        eval_model_name="claude",
        feedback_name="base",
        threshold_quality_metric="abstain_accuracy",
        **kwargs,
    ):
        self.eval_model_name = eval_model_name
        self.feedback_name = feedback_name
        self.threshold_quality_metric = threshold_quality_metric
        self.image_threshold = None
        self.answer_threshold = None
        super().__init__(**kwargs)
        # if len(glob.glob(f'./output/{self.dataset_name}/model/{self.eval_model_name}_{self.feedback_name}/*.safetensors')) == 0:
        #     self.generate_feedback()
        #     self.filter_feedback()
        #     self.finetune()

    @property
    def output_name(self):
        return self.method_name + "_" + self.feedback_name + "_" + self.split

    def generate_grounding(self):
        with open(f"instruction/generate_description.txt", encoding="utf-8") as f:
            instruction = "".join(f.readlines())

        for model_key, model_name in JURY_MODEL_KEY_TO_NAME.items():
            target_attr = f"image_desc_{model_key}"

            if all(
                target_attr in data and data[target_attr] != FAILED_TOKEN
                for data in self.dataset.get_split(self.split_flag)
            ):
                print(f"Generate grounding already done for model: {model_name}")
                continue

            query_list = self.dataset.customize_queries(
                key=target_attr,
                dtype=model_name_to_dtype(model_name),
                split_flag=self.split_flag,
                sample_size=1,
                instruction_func=lambda data, idx: instruction,
                include_option=False,
            )
            _, responses = batch_query_llm(
                query_list,
                model_name=model_name,
                temperature=0.0,
            )
            self._check_responses(responses, target_attr, model_name)
            for idx, data in enumerate(self.dataset.get_split(self.split_flag)):
                data[target_attr] = responses[idx]

            self.save_response()

    def generate_ground_truth(self):
        required_keys = [
            f"image_desc_{model_key}" for model_key in JURY_MODEL_KEY_TO_NAME.keys()
        ]
        for key in required_keys:
            if not all(key in data for data in self.dataset.get_split(self.split_flag)):
                raise ValueError(
                    f"Missing descriptions from {key} in the dataset. Ensure they are generated first."
                )
        with open(f"instruction/generate_ground_truth.txt", encoding="utf-8") as f:
            instruction_template = "".join(f.readlines())

        query_list = self.dataset.customize_queries(
            key="image_desc_gt",
            dtype=model_name_to_dtype(JUDGE_MODEL_NAME),
            split_flag=self.split_flag,
            sample_size=1,
            instruction_func=lambda data, i: instruction_template.format(
                *[data[key] for key in required_keys],
                (
                    IDX_TO_LETTER[
                        data["sampled_idxs"][i].index(data["correct_answer_idx"])
                    ]
                    if self.dataset.is_multichoice_dataset()
                    else data["answers"][0]
                ),
            ),
            include_image=False,
        )
        _, responses = batch_query_llm(
            query_list,
            model_name=JUDGE_MODEL_NAME,
            temperature=0.0,
            max_new_tokens=None,
        )
        self._check_responses(responses, f"image_desc_gt", JUDGE_MODEL_NAME)
        for idx, data in enumerate(self.dataset.get_split(self.split_flag)):
            data["image_desc_gt"] = responses[idx]

        self.save_response()

    def extract_ground_truth(self):
        with open(f"instruction/extract_ground_truth.txt", encoding="utf-8") as f:
            instruction = "".join(f.readlines())

        query_list = self.dataset.customize_queries(
            key="extracted_image_desc_gt",
            dtype=model_name_to_dtype(JUDGE_MODEL_NAME),
            split_flag=self.split_flag,
            sample_size=1,
            instruction_func=lambda data, i: instruction.format(data["image_desc_gt"]),
            include_image=False,
            include_option=False,
            include_question=False,
        )
        _, responses = batch_query_llm(
            query_list,
            model_name=JUDGE_MODEL_NAME,
            temperature=0.0,
            max_new_tokens=None,
        )
        self._check_responses(responses, f"extracted_image_desc_gt", JUDGE_MODEL_NAME)
        for idx, data in enumerate(self.dataset.get_split(self.split_flag)):
            data["extracted_image_desc_gt"] = responses[idx]
        self.save_response()

    def generate_sample_image_desc(self, sample_size=None, temperature=0.5):
        # self.generate_grounding()
        # self.generate_ground_truth()
        # self.extract_ground_truth()
        if sample_size is None:
            sample_size = self.sample_size

        with open("instruction/generate_description.txt", encoding="utf-8") as f:
            description_instruction = "".join(f.readlines())

        dtype = model_name_to_dtype(self.model_name)
        query_list = self.dataset.customize_queries(
            key="sample_image_desc",
            dtype=dtype,
            split_flag=self.split_flag,
            sample_size=sample_size,
            instruction_func=lambda data, idx: description_instruction,
            include_option=False,
        )
        _, description_responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            peft_dir=self.peft_dir,
            temperature=temperature,
            batch_size=sample_size,
        )

        for idx, data in enumerate(self.dataset.get_split(self.split_flag)):
            data["sample_image_desc"] = description_responses[
                idx * sample_size : (idx + 1) * sample_size
            ]

        self.save_response()

    def generate_image_desc_confidence(self):
        with open("instruction/description_score.txt", encoding="utf-8") as f:
            description_score_instruction = "".join(f.readlines())

        query_list = self.dataset.customize_queries(
            key="image_desc_confidence_scores",
            dtype=model_name_to_dtype(JUDGE_MODEL_NAME),
            split_flag=self.split_flag,
            sample_size=1,
            instruction_func=lambda data, i: description_score_instruction.format(
                data["extracted_image_desc_gt"],
                "\n".join(
                    [
                        f"Sample {j+1}: {desc}"
                        for j, desc in enumerate(data["sample_image_desc"])
                    ]
                ),
            ),
            include_image=False,
            include_option=False,
        )
        _, score_responses = batch_query_llm(
            query_list,
            model_name=JUDGE_MODEL_NAME,
            temperature=0,
            max_new_tokens=None,
        )
        self._check_responses(
            score_responses, f"image_desc_confidence_scores", JUDGE_MODEL_NAME
        )

        for idx, data in enumerate(self.dataset.get_split(self.split_flag)):
            score_response = score_responses[idx]
            if score_response == FAILED_TOKEN:
                data["image_desc_confidence_scores"] = FAILED_TOKEN
                data["image_desc_confidence"] = FAILED_TOKEN
                continue

            split_score_text = score_response.split("Score: ")[-1].split(",")
            scores = []

            if len(split_score_text) == self.sample_size:
                for s in split_score_text:
                    _s = s.strip()
                    try:
                        scores.append(int(_s))
                    except ValueError:
                        print(f"[id: {data['id']}] Generated score not int: {_s}")
                        scores.append(0)
            else:
                print(
                    f"[id: {data['id']}] There were {self.sample_size} sample image desc, but only {len(split_score_text)} confidence scores were generated."
                )
                scores = [0] * self.sample_size

            data["image_desc_confidence_scores"] = scores
            data["image_desc_confidence"] = float(np.average(scores))

        self.save_response()

    def generate_answer_confidence(self):
        for data in self.dataset.get_split(self.split_flag):
            _, correct_ratio = count_correct_answers(data)
            data["answer_confidence"] = correct_ratio

        self.save_response()

    def image_confidence_analysis(self):
        image_confidences = []
        for data in self.dataset.get_split(self.split_flag):
            if "image_desc_confidence" in data:
                image_confidences.append(data["image_desc_confidence"])

        plt.figure(figsize=(10, 6))
        plt.hist(
            image_confidences, bins=10, range=(0, 10), edgecolor="black", alpha=0.7
        )
        plt.title(f"Image Confidence Distribution for {self.model_name}")
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.xticks(range(0, 11))  # x-axis ticks from 0 to 10
        plt.grid(True)
        # plt.axvline(x=4.9, color='red', linestyle='--', linewidth=2)
        # plt.text(4.9, plt.ylim()[1] * 0.9, 'Threshold', color='red', ha='center')
        plt.savefig("image_desc_confidence.png", bbox_inches="tight")
        plt.close()

    def _threshold_config_filepath(self):
        return f"./config/r_tuning/{self.dataset_name}/{self.model_name}/val_thresholds_{self.feedback_name}_{QUALITY_METRIC_TO_POSTFIX[self.threshold_quality_metric]}.json"

    def determine_optimal_image_threshold(self):
        config_filepath = self._threshold_config_filepath()

        # if split is not "v", load from existing config
        if self.split_flag != "v":
            assert os.path.exists(
                config_filepath
            ), "If split 'v' is not specified, the threshold config file built from split 'v' must exist."

            self._load_optimal_threshold("image")
            return

        # if split is "v", calculate the threshold
        pred_flags = []
        label_flags = []
        prob_flags = []
        for data in self.dataset.get_split("v"):
            label_flags += [
                idxs.index(data["correct_answer_idx"]) for idxs in data["sampled_idxs"]
            ]
            pred_flags += data["extracted_responses"]
            for _ in range(len(data["extracted_responses"])):
                prob_flags.append(
                    data["image_desc_confidence"] / 10.0
                    if data["image_desc_confidence"] != FAILED_TOKEN
                    else FAILED_TOKEN
                )

        self.image_threshold = find_optimal_abstention_threshold(
            pred_flags,
            label_flags,
            prob_flags,
            quality_metric=self.threshold_quality_metric,
        )
        self.image_threshold = round(self.image_threshold, 8)
        print("threshold for image description calculated:", self.image_threshold)

        # save to config file
        if os.path.exists(config_filepath):
            threshold_config = load_from_json(config_filepath, print_msg=False)
        else:
            threshold_config = {}

        threshold_config["image_threshold"] = self.image_threshold
        dump_to_json(config_filepath, threshold_config)

    def construct_dataset(self):
        if self.split_flag == "e":
            raise ValueError("Should not construct dataset with test split")

        random.seed(get_seed())
        dataset = []

        with open(f"instruction/generate_description.txt", encoding="utf-8") as f:
            image_desc_instruction = "".join(f.readlines())

        prefix_boundary_dict = self._get_prefix_boundaries()
        visual_boundary_alpha, visual_boundary_beta = prefix_boundary_dict["visual"]
        answer_boundary_alpha, answer_boundary_beta = prefix_boundary_dict["answer"]

        for data in tqdm(
            self.dataset.get_split(self.split_flag),
            desc="Constructing Training Data (Two Stage)",
        ):
            # visual prefix
            if data["image_desc_confidence"] < visual_boundary_alpha:
                visual_prefix = random.choice(VISUAL_UNKNOWN_PREFIXES)
            elif data["image_desc_confidence"] < visual_boundary_beta:
                visual_prefix = random.choice(VISUAL_UNCERTAIN_PREFIXES)
            else:
                visual_prefix = random.choice(VISUAL_CONFIDENT_PREFIXES)

            visual_response = (
                f"{visual_prefix} {random.choice(data['sample_image_desc'])}"
            )

            # answer prefix
            correct_count = count_correct_answers(data)[0]
            if correct_count < answer_boundary_alpha:
                prefix = random.choice(ANSWER_UNKNOWN_PREFIXES)
            elif correct_count < answer_boundary_beta:
                prefix = random.choice(ANSWER_UNCERTAIN_PREFIXES)
            else:
                prefix = random.choice(ANSWER_CONFIDENT_PREFIXES)

            selected_response_idx = random.randint(0, len(data["responses"]) - 1)
            selected_response = data["responses"][selected_response_idx]
            new_response = f"{prefix} {selected_response}"

            first_round_messages = [
                {
                    "role": "user",
                    "content": build_finetune_user_content(
                        f"### Question:\n{data['question']}\n{image_desc_instruction}",
                        self.model_name,
                    ),
                },
                {
                    "role": "assistant",
                    "content": build_finetune_assistant_content(visual_response),
                },
            ]

            second_round_messages = [
                {
                    "role": "user",
                    "content": build_finetune_user_content(
                        data["queries"][selected_response_idx], self.model_name
                    ),
                },
                {
                    "role": "assistant",
                    "content": build_finetune_assistant_content(new_response),
                },
            ]

            messages = first_round_messages + second_round_messages
            image_path = f"./output/{self.dataset_name}/dataset/{self.split}_image/{data['id']}.png"
            dataset.append({"messages": messages, "image_path": image_path})

        output_file = f"./output/{self.dataset_name}/training_data/{self.model_name}/{self.output_name}.json"
        dump_to_json(output_file, dataset)

    def _get_prefix_boundaries(self) -> Dict[str, tuple]:
        best_alpha_beta = {"visual": (-1, -1), "answer": (-1, -1)}
        best_bucket_distribution = {}
        max_entropy = -1

        def bucket_key(visual_conf_level: str, answer_conf_level: str) -> str:
            return f"visual-{visual_conf_level}_answer-{answer_conf_level}"

        def init_bucket_dict() -> dict:
            bucket_dict = {}
            for visual_conf_level in CONFIDENCE_LEVELS:
                for answer_conf_level in CONFIDENCE_LEVELS:
                    bucket_dict[bucket_key(visual_conf_level, answer_conf_level)] = 0

            return bucket_dict

        for visual_alpha in range(1, 9):
            for visual_beta in range(visual_alpha + 1, 10):
                for answer_a in range(1, 8):
                    answer_alpha = answer_a - 0.5  # 0.5 - 6.5
                    for answer_b in range(answer_a + 1, 9):
                        answer_beta = answer_b - 0.5  # 1.5 - 7.5

                        bucket_to_count = init_bucket_dict()
                        for data in self.dataset.get_split(self.split_flag):
                            image_confidence = data["image_desc_confidence"]
                            if image_confidence < visual_alpha:
                                vis_conf_level = CONFIDENCE_LEVELS[0]
                            elif image_confidence < visual_beta:
                                vis_conf_level = CONFIDENCE_LEVELS[1]
                            else:
                                vis_conf_level = CONFIDENCE_LEVELS[2]

                            correct_count = count_correct_answers(data)[0]
                            if correct_count < answer_alpha:
                                ans_conf_level = CONFIDENCE_LEVELS[0]
                            elif correct_count < answer_beta:
                                ans_conf_level = CONFIDENCE_LEVELS[1]
                            else:
                                ans_conf_level = CONFIDENCE_LEVELS[2]

                            bucket_to_count[
                                bucket_key(vis_conf_level, ans_conf_level)
                            ] += 1

                        probs_each_bucket = [
                            count / len(self.dataset.get_split(self.split_flag))
                            for count in bucket_to_count.values()
                        ]
                        _entropy = entropy(probs_each_bucket)
                        if _entropy > max_entropy:
                            max_entropy = _entropy
                            best_alpha_beta = {
                                "visual": (visual_alpha, visual_beta),
                                "answer": (answer_alpha, answer_beta),
                            }
                            best_bucket_distribution = bucket_to_count

        # logging
        dump_to_json(
            f"./output/{self.dataset_name}/training_data/{self.model_name}/distribution/{self.output_name}.json",
            {
                "boundaries": {k: list(v) for k, v in best_alpha_beta.items()},
                "distribution": best_bucket_distribution,
                "entropy": max_entropy,
            },
        )

        print(
            f"Visual Prefix Boundaries: {best_alpha_beta['visual']} | Answer Prefix Boundaries: {best_alpha_beta['answer']}"
        )
        return best_alpha_beta

    def construct_dataset_only_answer(self):
        if self.split_flag == "e":
            raise ValueError("Should not construct dataset with test split")

        random.seed(get_seed())
        alpha, beta = self._get_answer_prefix_boundaries()

        dataset = []
        for data in tqdm(
            self.dataset.get_split(self.split_flag),
            desc="Constructing Training Data (Only Answers)",
        ):
            correct_count = count_correct_answers(data)[0]
            if correct_count < alpha:
                prefix = random.choice(ANSWER_UNKNOWN_PREFIXES)
            elif correct_count < beta:
                prefix = random.choice(ANSWER_UNCERTAIN_PREFIXES)
            else:
                prefix = random.choice(ANSWER_CONFIDENT_PREFIXES)

            selected_response_idx = random.randint(0, len(data["responses"]) - 1)
            selected_response = data["responses"][selected_response_idx]
            messages = [
                {
                    "role": "user",
                    "content": build_finetune_user_content(
                        data["queries"][selected_response_idx], self.model_name
                    ),
                },
                {
                    "role": "assistant",
                    "content": build_finetune_assistant_content(
                        f"{prefix} {selected_response}"
                    ),
                },
            ]

            image_path = f"./output/{self.dataset_name}/dataset/{self.split}_image/{data['id']}.png"
            dataset.append({"messages": messages, "image_path": image_path})

        output_file = f"./output/{self.dataset_name}/training_data/{self.model_name}/only_answer/{self.output_name}.json"
        dump_to_json(output_file, dataset)

    def _get_answer_prefix_boundaries(self) -> Tuple[float, float]:
        # 1-D boundary selection on answer confidences (CRes)
        best_alpha_beta = (-1, -1)
        best_bucket_distribution = {}
        max_entropy = -1

        for a in range(1, 8):
            alpha = a - 0.5  # 0.5 - 6.5
            for b in range(a + 1, 9):
                beta = b - 0.5  # 1.5 - 7.5
                bucket_to_count = {conf_level: 0 for conf_level in CONFIDENCE_LEVELS}
                for data in self.dataset.get_split(self.split_flag):
                    correct_count = count_correct_answers(data)[0]  # CRes
                    if correct_count < alpha:
                        ans_conf_level = CONFIDENCE_LEVELS[0]
                    elif correct_count < beta:
                        ans_conf_level = CONFIDENCE_LEVELS[1]
                    else:
                        ans_conf_level = CONFIDENCE_LEVELS[2]

                    bucket_to_count[ans_conf_level] += 1

                probs_each_bucket = [
                    count / len(self.dataset.get_split(self.split_flag))
                    for count in bucket_to_count.values()
                ]
                _entropy = entropy(probs_each_bucket)
                if _entropy > max_entropy:
                    max_entropy = _entropy
                    best_alpha_beta = (alpha, beta)
                    best_bucket_distribution = bucket_to_count

        # logging
        dump_to_json(
            f"./output/{self.dataset_name}/training_data/{self.model_name}/only_answer/distribution/{self.output_name}.json",
            {
                "boundaries": list(best_alpha_beta),
                "distribution": best_bucket_distribution,
                "entropy": max_entropy,
            },
        )

        print(f"Answer Prefix Boundaries: {best_alpha_beta}")
        return best_alpha_beta

    def construct_dataset_one_stage(self):
        raise NotImplementedError(
            "Code not updated yet. Should update code based on 'construct_dataset()'"
        )

        if self.split_flag != "t":
            raise ValueError("Able to construct dataset with only split flag 't'")

        random.seed(get_seed())
        dataset = []

        with open(f"instruction/generate_description.txt", encoding="utf-8") as f:
            image_desc_instruction = "".join(f.readlines())

        for data in tqdm(
            self.dataset.get_split(self.split_flag),
            desc="Constructing Training Data (One Stage)",
        ):
            assert data["image_desc_confidence"] != FAILED_TOKEN

            if data["image_desc_confidence"] <= 3.0:  # approx 1/3, variable
                visual_prefix = random.choice(VISUAL_UNKNOWN_PREFIXES)
            elif data["image_desc_confidence"] <= 7.0:  # approx 1/3, variable
                visual_prefix = random.choice(VISUAL_UNCERTAIN_PREFIXES)
            else:
                visual_prefix = random.choice(VISUAL_CONFIDENT_PREFIXES)

            correct_count, _ = count_correct_answers(data)

            if correct_count == 0:  # Unknown
                answer_prefix = random.choice(ANSWER_UNKNOWN_PREFIXES)
            elif 1 <= correct_count <= 6:  # Uncertain
                answer_prefix = random.choice(ANSWER_UNCERTAIN_PREFIXES)
            elif 7 <= correct_count <= 8:  # Confident
                answer_prefix = random.choice(ANSWER_CONFIDENT_PREFIXES)

            selected_response = random.choice(data["responses"])
            combined_response = f"{visual_prefix} {answer_prefix} {selected_response}"

            messages = [
                {
                    "content": f"### Question\n {data['question']}\n{image_desc_instruction}",
                    "role": "user",
                },
                {"content": combined_response, "role": "assistant"},
            ]

            for message in messages:
                if "<image 1>" in message["content"]:
                    message["content"] = message["content"].replace(
                        "<image 1>", "<image>"
                    )

            images = [f"{self.dataset_name}/{data['id']}.png"]
            dataset.append({"messages": messages, "images": images})

        output_file = f"./output/{self.dataset_name}/training_data/{self.model_name}/{self.output_name}.json"
        dump_to_json(output_file, dataset)

    def _load_optimal_threshold(self, typ: str):
        assert typ in ["image", "answer"]

        threshold_config = load_from_json(
            self._threshold_config_filepath(), print_msg=False
        )
        if f"{typ}_threshold" not in threshold_config:
            raise ValueError(f"{typ}_threshold not written in config file")

        threshold = threshold_config[f"{typ}_threshold"]
        if typ == "image":
            self.image_threshold = threshold
        else:
            self.answer_threshold = threshold

        print(f"threshold for {typ} loaded:", threshold)

    def determine_optimal_answer_threshold(self):
        config_filepath = self._threshold_config_filepath()

        # if split is not "v", load from existing config
        if self.split_flag != "v":
            assert os.path.exists(
                config_filepath
            ), "If split 'v' is not specified, the threshold config file built from split 'v' must exist."

            self._load_optimal_threshold("answer")
            return

        # if split is "v", calculate the threshold
        pred_flags = []
        label_flags = []
        prob_flags = []
        for data in self.dataset.get_split("v"):
            label_flags += [
                idxs.index(data["correct_answer_idx"]) for idxs in data["sampled_idxs"]
            ]
            pred_flags += data["extracted_responses"]
            prob_flags += self._prob_flags_for_answer_threshold_calculation(data)

        self.answer_threshold = find_optimal_abstention_threshold(
            pred_flags,
            label_flags,
            prob_flags,
            quality_metric=self.threshold_quality_metric,
        )
        self.answer_threshold = round(self.answer_threshold, 8)
        print("threshold for answer calculated:", self.answer_threshold)

        # save to config file
        if os.path.exists(config_filepath):
            threshold_config = load_from_json(config_filepath, print_msg=False)
        else:
            threshold_config = {}

        threshold_config["answer_threshold"] = self.answer_threshold
        dump_to_json(config_filepath, threshold_config)

    def _prob_flags_for_answer_threshold_calculation(self, data):
        return [data["answer_confidence"]] * len(data["extracted_responses"])

    def generate_greedy_image_desc(self, target_attr: str):
        with open("instruction/generate_description.txt", encoding="utf-8") as f:
            description_instruction = "".join(f.readlines())

        query_list = self.dataset.customize_queries(
            key=target_attr,
            dtype=model_name_to_dtype(self.model_name),
            split_flag=self.split_flag,
            sample_size=1,
            instruction_func=lambda data, idx: description_instruction,
            include_option=False,
        )

        _, description_responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            peft_dir=self.peft_dir,
            temperature=0.0,
            batch_size=1,
        )

        for idx, data in enumerate(self.dataset.get_split(self.split_flag)):
            data[target_attr] = description_responses[idx]

        self.save_response()

    def abstain(self):
        self.abstain_image_desc()
        self.abstain_image_desc_confidence()
        self.abstain_answer()
        self.abstain_answer_confidence()
        self.abstain_make_decisions()

    def abstain_image_desc(self):
        # 1. generate image desc with abstain
        if all(
            "greedy_image_desc" in data
            for data in self.dataset.get_split(self.split_flag)
        ):
            print("[INFO] Greedy Image Desc already exists. Using the existing ones.")
            for data in self.dataset.get_split(self.split_flag):
                data["abstain_image_desc"] = data["greedy_image_desc"]

            self.save_response()
        else:
            self.generate_greedy_image_desc(target_attr="abstain_image_desc")

    def abstain_image_desc_confidence(self):
        with open("instruction/description_score.txt", encoding="utf-8") as f:
            description_score_instruction = "".join(f.readlines())

        query_list = self.dataset.customize_queries(
            key="abstain_image_desc_confidence",
            dtype=model_name_to_dtype(JUDGE_MODEL_NAME),
            split_flag=self.split_flag,
            sample_size=1,
            instruction_func=lambda data, i: description_score_instruction.format(
                data["extracted_image_desc_gt"], data["abstain_image_desc"]
            ),
            include_image=False,
            include_option=False,
        )

        _, score_responses = batch_query_llm(
            query_list,
            model_name=JUDGE_MODEL_NAME,
            temperature=0,
            max_new_tokens=None,
        )
        self._check_responses(
            score_responses, f"abstain_image_desc_confidence", JUDGE_MODEL_NAME
        )

        for idx, data in enumerate(self.dataset.get_split(self.split_flag)):
            score_response = score_responses[idx]
            if score_response == FAILED_TOKEN:
                data["abstain_image_desc_confidence"] = FAILED_TOKEN
                continue

            try:
                scores = [
                    int(s.strip())
                    for s in score_response.split("Score: ")[-1].split(",")
                ]
                data["abstain_image_desc_confidence"] = scores[-1]
            except Exception as e:
                print(f"Error parsing response: {score_response}\n{e}")
                data["abstain_image_desc_confidence"] = 0

        self.save_response()

    def abstain_answer(self):
        # 2. Question Answering with abstain
        query_list, _ = self.dataset.form_queries(
            dtype=model_name_to_dtype(self.model_name),
            split_flag=self.split_flag,
            # NOTE: uses the first of "sampled_idxs". CAUTION: if this is to be changed, look at method.utils.extract_multichoice_answers
            sample_size=1,
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

    def abstain_answer_confidence(self):
        # requery to get the confidence on the generated answer
        with open("instruction/abstain_requery_instruction.txt", encoding="utf-8") as f:
            requery_instruction = "".join(f.readlines())

        requery_list = self.dataset.customize_queries(
            key="abstain_answer_confidence",
            dtype=model_name_to_dtype(JUDGE_MODEL_NAME),
            split_flag=self.split_flag,
            sample_size=1,
            instruction_func=lambda data, i: requery_instruction.format(
                response=data["abstain_response"]
            ),
            include_question=False,
            include_image=False,
            include_option=False,
        )

        _, requery_responses = batch_query_llm(
            requery_list,
            model_name=JUDGE_MODEL_NAME,
            temperature=0.0,
            max_new_tokens=None,
        )
        self._check_responses(
            requery_responses, f"abstain_answer_confidence", JUDGE_MODEL_NAME
        )

        for i, data in enumerate(self.dataset.get_split(self.split_flag)):
            requery_response = requery_responses[i]
            if requery_response == FAILED_TOKEN:
                data["abstain_answer_confidence"] = FAILED_TOKEN
                continue

            confidence_str = requery_response.strip().split("Confidence: ")[-1]
            try:
                data["abstain_answer_confidence"] = int(confidence_str)
            except (ValueError, IndexError):
                data["abstain_answer_confidence"] = 0

        self.save_response()

    def abstain_make_decisions(self):
        if self.image_threshold is None:
            self._load_optimal_threshold("image")

        if self.answer_threshold is None:
            self._load_optimal_threshold("answer")

        for data in self.dataset.get_split(self.split_flag):
            data["image_desc_abstain"] = (
                (data["abstain_image_desc_confidence"] / 10.0) < self.image_threshold
                if data["abstain_image_desc_confidence"] != FAILED_TOKEN
                else True
            )

            data["answer_abstain"] = (
                (data["abstain_answer_confidence"] / 10.0) < self.answer_threshold
                if data["abstain_answer_confidence"] != FAILED_TOKEN
                else True
            )

            # make the final abstain decision based on 2 stages (image desc & answer)
            data["abstain"] = [data["image_desc_abstain"] or data["answer_abstain"]]

        self.save_response()

    def _check_responses(self, responses, attribute, model_name):
        """
        Check the generated respones, and notify to Slack if there are any failed responses.
        """
        check_generate_failure(
            responses,
            attribute,
            model_name,
            f"./output/{self.dataset_name}/response/{self.model_name}/{self.output_name}.json",
        )
