from method.calibration import Calibration
from common.const import ABSTAIN_TOKEN, FAILED_TOKEN


class WeightedConsistency(Calibration):
    def __init__(
        self,
        confidence_func=lambda data: [
            c if c != FAILED_TOKEN else 0.0 for c in data["confidence"]
        ],
        **kwargs
    ):
        # confidence_func=lambda data: [1.0 if abstain is False else 0.0 for abstain in data['abstain']]
        super().__init__(**kwargs)
        for data in self.dataset.get_split("v"):
            if "ori_confidence" not in data:
                data["ori_confidence"] = confidence_func(data)
                if "confidence" in data:
                    del data["confidence"]
                if "abstain" in data:
                    del data["abstain"]
        for data in self.dataset.get_split(self.split_flag):
            if "ori_confidence" not in data:
                data["ori_confidence"] = confidence_func(data)
                if "confidence" in data:
                    del data["confidence"]
                if "abstain" in data:
                    del data["abstain"]

    def _generate_confidence(self):
        for data in self.dataset.get_split(self.split_flag):
            confidence_list = []
            extracted_responses = [
                idxs[e] if (e != ABSTAIN_TOKEN and e != FAILED_TOKEN) else e
                for idxs, e in zip(data["sampled_idxs"], data["extracted_responses"])
            ]
            for i in range(self.sample_size):
                confidence = 0.0
                for j in range(self.sample_size):
                    if extracted_responses[j] == extracted_responses[i]:
                        confidence += (
                            data["ori_confidence"][j]
                            if data["ori_confidence"][j] != FAILED_TOKEN
                            else 0.0
                        )
                confidence_list.append(
                    confidence / (sum(data["ori_confidence"]) + 1e-6)
                )
            data["confidence"] = confidence_list

    def abstain(self):
        self.calculate_confidence()
        self.determine_optimal_threshold()
        for data in self.dataset.get_split(self.split_flag):
            data["abstain"] = [
                c < self.threshold if c != FAILED_TOKEN else FAILED_TOKEN
                for c in data["confidence"]
            ]
        self.method_name = "weighted_" + self.method_name
        self.save_response()
        self.method_name = self.method_name[9:]
