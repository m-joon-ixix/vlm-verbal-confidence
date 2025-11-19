import os
from abc import abstractmethod

from method.base import BaseMethod
from method.utils import find_optimal_abstention_threshold
from common.const import FAILED_TOKEN
from dataset.dataset_utils import load_json_data_with_image


class Calibration(BaseMethod):
    def __init__(
        self,
        dataset_name,
        load_from_exist=True,
        threshold_quality_metric="abstain_accuracy",
        **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name, load_from_exist=load_from_exist, **kwargs
        )
        self.threshold = 0.0
        self.threshold_quality_metric = threshold_quality_metric
        self.dataset.get_split("v").clear()

        if load_from_exist and os.path.exists(
            f"./output/{self.dataset_name}/response/{self.model_name}/{self.method_name}_val.json"
        ):
            load_json_data_with_image(
                self.dataset.get_split("v"),
                f"./output/{self.dataset_name}/response/{self.model_name}",
                f"./output/{self.dataset_name}/dataset",
                f"{self.method_name}_val",
            )
        else:
            load_json_data_with_image(
                self.dataset.get_split("v"),
                (
                    f"./output/{self.dataset_name}/response/{self.model_name}"
                    if self.method_name != "origin"
                    else f"./output/{self.dataset_name}/dataset"
                ),
                f"./output/{self.dataset_name}/dataset",
                ("origin_" if self.method_name != "origin" else "") + "val",
            )

    def calculate_confidence(self):
        if not all(
            ["confidence" in data for data in self.dataset.get_split(self.split_flag)]
        ):
            self._generate_confidence()

        if not all(["confidence" in data for data in self.dataset.get_split("v")]):
            temp_split_flag = self.split_flag
            self.split_flag = "v"
            self._generate_confidence()
            self.split_flag = temp_split_flag

    @abstractmethod
    def _generate_confidence(self):
        pass

    def determine_optimal_threshold(self):
        pred_flags = []
        label_flags = []
        prob_flags = []
        for data in self.dataset.get_split("v"):
            label_flags += [
                idxs.index(data["correct_answer_idx"]) for idxs in data["sampled_idxs"]
            ]
            pred_flags += data["extracted_responses"]
            prob_flags += data["confidence"]

        self.threshold = find_optimal_abstention_threshold(
            pred_flags,
            label_flags,
            prob_flags,
            quality_metric=self.threshold_quality_metric,
        )
        print(f"[{self.method_name}] threshold from 'val' split: {self.threshold}")

    def abstain(self):
        self.calculate_confidence()
        self.determine_optimal_threshold()

        for data in self.dataset.get_split(self.split_flag):
            data["abstain"] = [
                c < self.threshold if c != FAILED_TOKEN else FAILED_TOKEN
                for c in data["confidence"]
            ]
        self.save_response()

        temp_split_flag = self.split_flag
        self.split_flag = "v"
        self.save_response()
        self.split_flag = temp_split_flag
