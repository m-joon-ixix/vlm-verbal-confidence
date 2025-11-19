import os
import torch
from typing import Callable
from abc import abstractmethod
from torch.utils.data import random_split

from dataset.dataset_utils import load_json_data_with_image, save_json_data
from common.random_utils import get_seed


class BaseDataset:
    dataset_name: str
    sample_size: int = 50
    extract_instruction: str
    extract_pattern: str
    post_process: Callable

    def __init__(
        self, dataset_name, sample_size=2000, load_from_exist=True, load_flag=""
    ):
        self.dataset = []
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        create_dataset = False
        if "t" in load_flag:
            if load_from_exist and os.path.exists(
                f"./output/{self.dataset_name}/dataset/train.json"
            ):
                load_json_data_with_image(
                    self.train_dataset,
                    f"./output/{self.dataset_name}/dataset",
                    f"./output/{self.dataset_name}/dataset",
                    "train",
                )
            else:
                create_dataset = True
        if "v" in load_flag:
            if load_from_exist and os.path.exists(
                f"./output/{self.dataset_name}/dataset/val.json"
            ):
                load_json_data_with_image(
                    self.val_dataset,
                    f"./output/{self.dataset_name}/dataset",
                    f"./output/{self.dataset_name}/dataset",
                    "val",
                )
            else:
                create_dataset = True
        if "e" in load_flag:
            if load_from_exist and os.path.exists(
                f"./output/{self.dataset_name}/dataset/test.json"
            ):
                load_json_data_with_image(
                    self.test_dataset,
                    f"./output/{self.dataset_name}/dataset",
                    f"./output/{self.dataset_name}/dataset",
                    "test",
                )
            else:
                create_dataset = True
        if create_dataset:
            self.load_dataset()
            self.split_dataset()
            self.save_dataset(save_flag="tve")

    def load_dataset(self):
        pass

    @abstractmethod
    def is_multichoice_dataset(self):
        pass

    @abstractmethod
    def form_queries(self, **kwargs):
        pass

    @abstractmethod
    def customize_queries(self, **kwargs):
        pass

    def split_dataset(self):
        train_dataset_size = round(len(self.dataset) * 0.8)
        val_dataset_size = round(len(self.dataset) * 0.1)
        torch.manual_seed(get_seed())
        torch.cuda.manual_seed(get_seed())
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [
                train_dataset_size,
                val_dataset_size,
                len(self.dataset) - train_dataset_size - val_dataset_size,
            ],
        )
        self.train_dataset = list(self.train_dataset)
        self.val_dataset = list(self.val_dataset)
        self.test_dataset = list(self.test_dataset)
        print(
            f"Splits created => Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )

    def save_dataset(self, save_flag=""):
        if "t" in save_flag:
            save_json_data(
                self.train_dataset,
                f"./output/{self.dataset_name}/dataset",
                f"./output/{self.dataset_name}/dataset",
                "train",
                save_image=True,
            )
        if "v" in save_flag:
            save_json_data(
                self.val_dataset,
                f"./output/{self.dataset_name}/dataset",
                f"./output/{self.dataset_name}/dataset",
                "val",
                save_image=True,
            )
        if "e" in save_flag:
            save_json_data(
                self.test_dataset,
                f"./output/{self.dataset_name}/dataset",
                f"./output/{self.dataset_name}/dataset",
                "test",
                save_image=True,
            )

    def get_split(self, split_flag, debug=False):
        if split_flag == "t":
            if debug and len(self.train_dataset) > 10:
                self.train_dataset = self.train_dataset[:10]
            return self.train_dataset
        elif split_flag == "v":
            if debug and len(self.val_dataset) > 10:
                self.val_dataset = self.val_dataset[:10]
            return self.val_dataset
        elif split_flag == "e":
            if debug and len(self.test_dataset) > 10:
                self.test_dataset = self.test_dataset[:10]
            return self.test_dataset
        else:
            raise NotImplementedError
