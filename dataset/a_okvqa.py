import random
import datasets

from dataset.mmmu import MmmuDataset
from common.const import DATASET_CACHE_DIR
from common.random_utils import get_seed


class AOkvqaDataset(MmmuDataset):
    def load_dataset(self):
        print(f"Loading Raw Dataset (dataset_name: {self.dataset_name})")

        self.dataset = []
        random.seed(get_seed())
        raw_dataset = datasets.load_dataset(
            "HuggingFaceM4/A-OKVQA", cache_dir=DATASET_CACHE_DIR
        )
        for data in raw_dataset["validation"]:
            self.dataset.append(
                {
                    "id": data["question_id"],
                    "image": data["image"],
                    "question": data["question"],
                    "options": data["choices"],
                    "correct_answer_idx": data["correct_choice_idx"],
                }
            )
        random.seed(get_seed())
        if 0 < self.sample_size < len(self.dataset):
            self.dataset = random.sample(self.dataset, self.sample_size)
        else:
            random.shuffle(self.dataset)
