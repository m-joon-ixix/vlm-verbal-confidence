import os
import random
from tqdm import tqdm

from dataset.dataset_utils import (
    load_custom_dataset,
    load_json_data_with_image,
    save_json_data,
)
from method.utils import clean_extracted_answers, extract_multichoice_answers
from common.const import ABSTAIN_TOKEN, FLAG_TO_SPLIT
from common.model_names import DType, model_name_to_dtype
from common.json_utils import dump_to_json
from common.random_utils import get_seed
from generate.form_query import (
    build_finetune_user_content,
    build_finetune_assistant_content,
)
from generate.common import batch_query_llm


class BaseMethod:
    method_name = "origin"

    def __init__(
        self,
        dataset_name,
        model_name,
        method_name=None,
        peft_dir=None,
        split_flag="t",
        response_sample_size=4,
        load_from_exist=True,
    ):
        assert len(split_flag) == 1
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.peft_dir = peft_dir
        if method_name is not None:
            self.method_name = method_name
        self.split_flag = split_flag
        assert response_sample_size > 0
        self.sample_size = response_sample_size
        self.dataset = load_custom_dataset(
            dataset_name, load_flag=self.split_flag, load_from_exist=True
        )

        if load_from_exist and os.path.exists(
            f"./output/{self.dataset_name}/response/{self.model_name}/{self.output_name}.json"
        ):
            load_json_data_with_image(
                self.dataset.get_split(self.split_flag),
                f"./output/{self.dataset_name}/response/{self.model_name}",
                f"./output/{self.dataset_name}/dataset",
                self.output_name,
            )
        else:
            load_json_data_with_image(
                self.dataset.get_split(self.split_flag),
                (
                    f"./output/{self.dataset_name}/response/{self.model_name}"
                    if self.method_name != "origin"
                    else f"./output/{self.dataset_name}/dataset"
                ),
                f"./output/{self.dataset_name}/dataset",
                ("origin_" if self.method_name != "origin" else "") + self.split,
            )

    @property
    def output_name(self):
        return self.method_name + "_" + self.split

    @property
    def split(self):
        return FLAG_TO_SPLIT[self.split_flag]

    def save_response(self):
        save_json_data(
            self.dataset.get_split(self.split_flag),
            f"./output/{self.dataset_name}/response/{self.model_name}",
            f"./output/{self.dataset_name}/dataset",
            self.output_name,
        )

    def generate_responses(self, prob=False):
        if not self.dataset.is_multichoice_dataset():
            # Open-Ended QA
            self.generate_open_ended_responses()
            return

        # Multi-Choice QA
        dtype = model_name_to_dtype(self.model_name)
        query_list, sampled_idxs = self.dataset.form_queries(
            dtype, self.split_flag, self.sample_size
        )
        probs, responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            temperature=0.0,
            peft_dir=self.peft_dir,
            # output_logits=prob
        )
        for i, data in enumerate(self.dataset.get_split(self.split_flag)):
            if dtype == DType.GEMINI:
                # gemini's query is different, extract from 'parts'
                data["queries"] = [
                    body[0]["parts"][0]["text"]
                    for body in query_list[
                        i * self.sample_size : (i + 1) * self.sample_size
                    ]
                ]
            else:  # extract from 'content' by default
                data["queries"] = [
                    body[0]["content"][0]["text"]
                    for body in query_list[
                        i * self.sample_size : (i + 1) * self.sample_size
                    ]
                ]
            data["sampled_idxs"] = sampled_idxs[
                i * self.sample_size : (i + 1) * self.sample_size
            ]
            data["responses"] = responses[
                i * self.sample_size : (i + 1) * self.sample_size
            ]
            if prob:
                data["probs"] = probs[i * self.sample_size : (i + 1) * self.sample_size]
        self.save_response()

    def generate_open_ended_responses(self):
        dtype = model_name_to_dtype(self.model_name)
        query_list = self.dataset.form_queries(dtype, self.split_flag, self.sample_size)

        _, responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            temperature=0.0 if self.sample_size == 1 else 0.5,
            peft_dir=self.peft_dir,
            batch_size=self.sample_size,
        )

        for i, data in enumerate(self.dataset.get_split(self.split_flag)):
            data["responses"] = responses[
                i * self.sample_size : (i + 1) * self.sample_size
            ]

        self.save_response()

    def parse_freeform_responses(
        self,
        input_key="responses",
        output_key="extracted_responses",
        instruction=None,
        extract_pattern=None,
        post_process=None,
        include_question=True,
        include_option=True,
        include_image=True,
    ):
        if instruction is None:
            instruction = self.dataset.extract_instruction
        if extract_pattern is None:
            extract_pattern = self.dataset.extract_pattern
        if post_process is None:
            post_process = self.dataset.post_process
        with open(f"instruction/{instruction}.txt", encoding="utf-8") as f:
            extract_instruction = "".join(f.readlines())
        query_list = self.dataset.customize_queries(
            key=output_key,
            dtype=model_name_to_dtype(self.model_name),
            split_flag=self.split_flag,
            sample_size=self.sample_size,
            instruction_func=lambda data, i: extract_instruction.format(
                data[input_key][i]
            ),
            include_question=include_question,
            include_option=include_option,
            include_image=include_image,
        )
        _, responses = batch_query_llm(
            query_list, model_name=self.model_name, temperature=0.0, max_new_tokens=16
        )
        for data in self.dataset.get_split(self.split_flag):
            data[output_key] = [responses[idx] for idx in data[output_key]]
        clean_extracted_answers(
            self.dataset.get_split(self.split_flag),
            extract_pattern,
            post_process,
            output_key,
        )
        self.save_response()

    def parse_structured_response(self, response_attr: str = "responses"):
        extract_multichoice_answers(
            self.dataset.get_split(self.split_flag), response_attr
        )
        self.save_response()

    def abstain(self):
        for data in self.dataset.get_split(self.split_flag):
            data["abstain"] = [r == ABSTAIN_TOKEN for r in data["extracted_responses"]]
        self.save_response()

    def construct_dataset(self):
        if self.split_flag == "e":
            raise ValueError("Should not construct dataset with test split")

        random.seed(get_seed())
        dataset = []

        for data in tqdm(
            self.dataset.get_split(self.split_flag),
            desc="Constructing Training Data (only answer)",
        ):
            # get indices where the extracted_response was correct
            correct_sample_indices = []
            for idx, (extracted_response, sampled_idx) in enumerate(
                zip(data["extracted_responses"], data["sampled_idxs"])
            ):
                if extracted_response == sampled_idx.index(data["correct_answer_idx"]):
                    correct_sample_indices.append(idx)

            # if there was no correct response, remove the example from training set
            if len(correct_sample_indices) == 0:
                continue

            selected_sample_idx = random.choice(correct_sample_indices)
            messages = [
                {
                    "role": "user",
                    "content": build_finetune_user_content(
                        data["queries"][selected_sample_idx], self.model_name
                    ),
                },
                {
                    "role": "assistant",
                    "content": build_finetune_assistant_content(
                        data["responses"][selected_sample_idx]
                    ),
                },
            ]

            image_path = f"./output/{self.dataset_name}/dataset/{self.split}_image/{data['id']}.png"
            dataset.append({"messages": messages, "image_path": image_path})

        output_file = f"./output/{self.dataset_name}/training_data/{self.model_name}/{self.output_name}.json"
        dump_to_json(output_file, dataset)
