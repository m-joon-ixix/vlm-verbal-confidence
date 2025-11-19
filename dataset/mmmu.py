import datasets
import random

from dataset.base import BaseDataset
from common.const import DATASET_CACHE_DIR, LETTER_TO_IDX
from common.model_names import DType
from common.random_utils import get_seed
from generate.form_query import form_multichoice_queries, customize_multichoice_query

CATEGORIES = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
    'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power',
    'Finance', 'Geography', 'History', 'Literature', 'Manage',
    'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music',
    'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]


class MmmuDataset(BaseDataset):
    def __init__(self, **kwargs):
        if kwargs["dataset_name"] == "MMMU":
            if "categories" in kwargs:
                self.categories = kwargs["categories"]
            else:
                self.categories = CATEGORIES
        self.instruction = "multi_choice_query"
        self.extract_instruction = "multi_choice_extract"
        self.extract_pattern = r"([A-Z])(?![A-Za-z])"  # newer version: method/utils.py => extract_multichoice_answers()
        self.post_process = lambda x: LETTER_TO_IDX[x]
        super().__init__(**kwargs)

    def load_dataset(self):
        print(f"Loading Raw Dataset (dataset_name: {self.dataset_name})")

        self.dataset = []
        for category in self.categories:
            raw_dataset = datasets.load_dataset(
                "MMMU/MMMU", category, cache_dir=DATASET_CACHE_DIR
            )
            for data in list(raw_dataset["dev"]) + list(raw_dataset["validation"]):
                if (
                    data["image_2"] is None
                    and data["question_type"] == "multiple-choice"
                ):
                    self.dataset.append(
                        {
                            "id": data["id"],
                            "image": data["image_1"],
                            "question": data["question"],
                            "options": eval(data["options"]),
                            "correct_answer_idx": LETTER_TO_IDX[data["answer"]],
                            "category": category,
                        }
                    )
        random.seed(get_seed())
        if 0 < self.sample_size < len(self.dataset):
            self.dataset = random.sample(self.dataset, self.sample_size)
        else:
            random.shuffle(self.dataset)

    def is_multichoice_dataset(self):
        return True

    def form_queries(
        self, dtype: DType, split_flag="t", sample_size=5, instruction_name=None
    ):
        if instruction_name is None:
            instruction_name = self.instruction
        return form_multichoice_queries(
            self.get_split(split_flag),
            sample_size,
            dtype,
            instruction_name=instruction_name,
        )

    def customize_queries(
        self,
        key,
        dtype: DType,
        split_flag="t",
        sample_size=5,
        skip_func=lambda x, idx: False,
        instruction_func=lambda x, idx: "",
        include_question=True,
        include_option=True,
        include_image=True,
    ):
        return customize_multichoice_query(
            self.get_split(split_flag),
            sample_size,
            key,
            dtype,
            skip_func,
            instruction_func,
            include_question,
            include_option,
            include_image,
        )
