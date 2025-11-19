import random
import datasets

from dataset.mmmu import MmmuDataset
from common.const import DATASET_CACHE_DIR
from common.random_utils import get_seed


class MathVistaDataset(MmmuDataset):
    def load_dataset(self):
        print(f"Loading Raw Dataset (dataset_name: {self.dataset_name})")

        self.dataset = []
        random.seed(get_seed())
        raw_dataset = datasets.load_dataset(
            "AI4Math/MathVista", cache_dir=DATASET_CACHE_DIR
        )
        for data in raw_dataset["testmini"]:
            if data["question_type"] == "multi_choice":
                self.dataset.append(
                    {
                        "id": data["pid"],
                        "image": data["decoded_image"],
                        "question": data["question"],
                        "options": data["choices"],
                        "correct_answer_idx": data["choices"].index(data["answer"]),
                    }
                )
        random.seed(get_seed())
        if 0 < self.sample_size < len(self.dataset):
            self.dataset = random.sample(self.dataset, self.sample_size)
        else:
            random.shuffle(self.dataset)
