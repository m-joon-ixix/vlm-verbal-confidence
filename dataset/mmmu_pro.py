import random
import datasets

from dataset.mmmu import MmmuDataset
from common.const import LETTER_TO_IDX, DATASET_CACHE_DIR
from common.random_utils import get_seed


class MmmuProDataset(MmmuDataset):
    def load_dataset(self):
        print(f"Loading Raw Dataset (dataset_name: {self.dataset_name})")

        self.dataset = []
        random.seed(get_seed())
        raw_dataset = datasets.load_dataset(
            "MMMU/MMMU_Pro", "standard (4 options)", cache_dir=DATASET_CACHE_DIR
        )
        for data in raw_dataset["test"]:
            if data["image_2"] is None:
                self.dataset.append(
                    {
                        "id": data["id"],
                        "image": data["image_1"],
                        "question": data["question"],
                        "options": eval(data["options"]),
                        "correct_answer_idx": LETTER_TO_IDX[data["answer"]],
                        "category": data["subject"],
                    }
                )
        random.seed(get_seed())
        if 0 < self.sample_size < len(self.dataset):
            self.dataset = random.sample(self.dataset, self.sample_size)
        else:
            random.shuffle(self.dataset)
