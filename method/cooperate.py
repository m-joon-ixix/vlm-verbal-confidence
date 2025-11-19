from method.base import BaseMethod
from generate.common import batch_query_llm
from common.model_names import model_name_to_dtype


class Cooperate(BaseMethod):
    jury_config: list
    method_name = "cooperate"

    def __init__(
        self,
        jury_model="gpt-4o",
        jury_expertise=("vision", "language", "all"),
        **kwargs,
    ):
        self.jury_config = []
        self.jury_model = jury_model
        self.jury_expertise = jury_expertise
        for e in jury_expertise:
            self.jury_config.append((jury_model, e, "pos"))
            self.jury_config.append((jury_model, e, "neg"))
        # TODO: debugging
        self.jury_config = [
            ("gpt-4o", "vision", ""),
            ("gpt-4o", "language", ""),
            ("gpt-4o", "all", ""),
        ]
        super().__init__(**kwargs)

    @property
    def output_name(self):
        return (
            self.method_name
            + "_"
            + self.jury_model
            + "_"
            + str("_".join(self.jury_expertise))
            + "_"
            + self.split
        )

    def generate_grounding(self):
        if not all(
            ["description" in data for data in self.dataset.get_split(self.split_flag)]
        ):
            with open(f"instruction/generate_description.txt", encoding="utf-8") as f:
                instruction = "".join(f.readlines())
            query_list = self.dataset.customize_queries(
                key=f"description",
                dtype=model_name_to_dtype(self.jury_model),
                split_flag=self.split_flag,
                sample_size=1,
                instruction_func=lambda data, idx: instruction,
                include_option=False,
            )
            _, responses = batch_query_llm(
                query_list,
                model_name=self.jury_model,
                temperature=0.0,
            )
            for data in self.dataset.get_split(self.split_flag):
                data["description"] = responses[data["description"][0]]
            self.save_response()

        if not all(
            ["knowledge" in data for data in self.dataset.get_split(self.split_flag)]
        ):
            with open(f"instruction/generate_knowledge.txt", encoding="utf-8") as f:
                instruction = "".join(f.readlines())
            query_list = self.dataset.customize_queries(
                key=f"knowledge",
                dtype=model_name_to_dtype(self.jury_model),
                split_flag=self.split_flag,
                sample_size=1,
                instruction_func=lambda data, idx: instruction,
                include_image=False,
            )
            _, responses = batch_query_llm(
                query_list,
                model_name=self.jury_model,
                temperature=0.0,
            )
            for data in self.dataset.get_split(self.split_flag):
                data["knowledge"] = responses[data["knowledge"][0]]
            self.save_response()

    def generate_vision_feedback(self):
        if not all(
            [
                "feedback_vision" in data
                for data in self.dataset.get_split(self.split_flag)
            ]
        ):
            with open(f"instruction/cooperate_vision.txt", encoding="utf-8") as f:
                instruction = "".join(f.readlines())
            query_list = self.dataset.customize_queries(
                key=f"feedback_vision",
                dtype=model_name_to_dtype(self.jury_model),
                split_flag=self.split_flag,
                sample_size=self.sample_size,
                instruction_func=lambda data, idx: instruction.format(
                    data["description"], data["responses"][idx]
                ),
            )
            _, responses = batch_query_llm(
                query_list,
                model_name=self.jury_model,
                temperature=0.0,
            )
            for data in self.dataset.get_split(self.split_flag):
                data["feedback_vision"] = [
                    responses[idx] for idx in data["feedback_vision"]
                ]
            self.save_response()

    def generate_language_feedback(self):
        if not all(
            [
                "feedback_language" in data
                for data in self.dataset.get_split(self.split_flag)
            ]
        ):
            with open(f"instruction/cooperate_language.txt", encoding="utf-8") as f:
                instruction = "".join(f.readlines())
            query_list = self.dataset.customize_queries(
                key=f"feedback_language",
                dtype=model_name_to_dtype(self.jury_model),
                split_flag=self.split_flag,
                sample_size=self.sample_size,
                instruction_func=lambda data, idx: instruction.format(
                    data["description"], data["knowledge"], data["responses"][idx]
                ),
                include_image=False,
            )
            _, responses = batch_query_llm(
                query_list,
                model_name=self.jury_model,
                temperature=0.0,
            )
            for data in self.dataset.get_split(self.split_flag):
                data["feedback_language"] = [
                    responses[idx] for idx in data["feedback_language"]
                ]
            self.save_response()

    def generate_plain_feedback(self):
        if not all(
            [
                "feedback_none" in data
                for data in self.dataset.get_split(self.split_flag)
            ]
        ):
            with open(f"instruction/cooperate_none.txt", encoding="utf-8") as f:
                instruction = "".join(f.readlines())
            query_list = self.dataset.customize_queries(
                key=f"feedback_none",
                dtype=model_name_to_dtype(self.jury_model),
                split_flag=self.split_flag,
                sample_size=self.sample_size,
                instruction_func=lambda data, idx: instruction.format(
                    data["responses"][idx]
                ),
            )
            _, responses = batch_query_llm(
                query_list,
                model_name=self.jury_model,
                temperature=0.0,
            )
            for data in self.dataset.get_split(self.split_flag):
                data["feedback_none"] = [
                    responses[idx] for idx in data["feedback_none"]
                ]
            self.save_response()

    def generate_general_feedback(self):
        if not all(
            ["feedback_all" in data for data in self.dataset.get_split(self.split_flag)]
        ):
            with open(f"instruction/cooperate_all.txt", encoding="utf-8") as f:
                instruction = "".join(f.readlines())
            query_list = self.dataset.customize_queries(
                key=f"feedback_all",
                dtype=model_name_to_dtype(self.jury_model),
                split_flag=self.split_flag,
                sample_size=self.sample_size,
                instruction_func=lambda data, idx: instruction.format(
                    data["description"], data["knowledge"], data["responses"][idx]
                ),
            )
            _, responses = batch_query_llm(
                query_list,
                model_name=self.jury_model,
                temperature=0.0,
            )
            for data in self.dataset.get_split(self.split_flag):
                data["feedback_all"] = [responses[idx] for idx in data["feedback_all"]]
            self.save_response()

    def abstain(self, key="vision"):
        self.generate_grounding()
        self.generate_vision_feedback()
        self.generate_language_feedback()
        self.generate_general_feedback()
        self.generate_plain_feedback()
        # def post_process(x):
        #     x = x.lower()
        #     if 'support' in x:
        #         return False
        #     elif 'refute' in x:
        #         return True
        #     elif 'abstain' in x:
        #         return ABSTAIN_TOKEN
        #     else:
        #         return FAILED_TOKEN
        # self.parse_freeform_responses(
        #     input_key=f'feedback_{key}',
        #     output_key=f'extracted_feedback_{key}',
        #     instruction='cooperate_extract',
        #     extract_pattern=r'(.*)',
        #     post_process=post_process,
        #     include_question=False,
        #     include_option=False,
        #     include_image=False,
        # )
        # for data in self.dataset.get_split(self.split_flag):
        #     data['abstain'] = data[f'extracted_feedback_{key}']
        for data in self.dataset.get_split(self.split_flag):
            data["abstain"] = []
            for i in range(self.sample_size):
                vote = 0
                for feedback in ["language", "vision", "all", "none"]:
                    vote += data[f"feedback_{feedback}"][i] is True
                data["abstain"].append(vote >= 2)
        self.save_response()
