from collections import Counter

from method.calibration import Calibration
from common.const import ABSTAIN_TOKEN, FAILED_TOKEN


class SelfConsistency(Calibration):
    method_name = "consistency"

    def _generate_confidence(self):
        for data in self.dataset.get_split(self.split_flag):
            responded_option_idxs = [
                idxs[e] if (e != ABSTAIN_TOKEN and e != FAILED_TOKEN) else e
                for idxs, e in zip(data["sampled_idxs"], data["extracted_responses"])
            ]

            option_idx_to_count = Counter(responded_option_idxs)

            # list of (float or FAILED_TOKEN)
            data["confidence"] = [
                (
                    option_idx_to_count[option_idx] / len(responded_option_idxs)
                    if option_idx != ABSTAIN_TOKEN and option_idx != FAILED_TOKEN
                    # consider ABSTAIN/FAILED TOKEN as sole responses
                    else 1 / len(responded_option_idxs)
                )
                for option_idx in responded_option_idxs
            ]
