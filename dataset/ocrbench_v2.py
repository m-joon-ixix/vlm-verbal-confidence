import random
import datasets

from dataset.base import BaseDataset
from generate.form_query import form_open_ended_queries, customize_open_ended_query
from common.const import DATASET_CACHE_DIR
from common.model_names import DType
from common.random_utils import get_seed


class OcrBenchV2Dataset(BaseDataset):
    def __init__(self, **kwargs):
        self.instruction = "open_ended_query"
        self.extract_instruction = "open_ended_extract"  # NOTE: not used
        self.extract_pattern = r"."  # FIXME
        self.post_process = lambda x: x  # FIXME
        super().__init__(**kwargs)

    def load_dataset(self):
        print(f"Loading Raw Dataset (dataset_name: {self.dataset_name})")

        self.dataset = []
        raw_dataset = datasets.load_dataset(
            "ling99/OCRBench_v2", split="test", cache_dir=DATASET_CACHE_DIR
        )
        for data in list(raw_dataset):
            if data["eval"] in ["None", "regression"]:
                self.dataset.append(
                    {
                        "id": data["id"],
                        "image": data["image"],
                        "question": data["question"],
                        "answers": data["answers"],
                    }
                )

        random.seed(get_seed())
        if 0 < self.sample_size < len(self.dataset):
            self.dataset = random.sample(self.dataset, self.sample_size)
        else:
            random.shuffle(self.dataset)

    def is_multichoice_dataset(self):
        return False

    def form_queries(
        self, dtype: DType, split_flag, sample_size=1, instruction_name=None
    ):
        if instruction_name is None:
            instruction_name = self.instruction

        return form_open_ended_queries(
            self.get_split(split_flag),
            sample_size,
            dtype,
            instruction_name=instruction_name,
        )

    # NOTE: unused arguments still exist to maintain a uniform interface with other dataset classes
    def customize_queries(
        self,
        key,
        dtype: DType,
        split_flag,
        sample_size=1,
        skip_func=lambda x, idx: False,
        instruction_func=lambda x, idx: "",
        include_question=True,
        include_option=True,
        include_image=True,
    ):
        return customize_open_ended_query(
            self.get_split(split_flag),
            sample_size,
            dtype,
            instruction_func,
            include_question,
            include_image,
        )
