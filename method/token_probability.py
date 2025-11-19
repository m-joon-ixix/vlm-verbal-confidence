from math import exp

from method.calibration import Calibration


class TokenProbability(Calibration):
    method_name = "prob"

    def _generate_confidence(self):
        # self.generate_responses(prob=True)
        # self.parse_structured_response()
        for data in self.dataset.get_split(self.split_flag):
            data["confidence"] = [exp(-p) for p in data["probs"]]
