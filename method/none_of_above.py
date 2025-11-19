from method.base import BaseMethod


class NoneOfAbove(BaseMethod):
    method_name = "noa"

    def generate_responses(self):
        for data in self.dataset.get_split(self.split_flag):
            if "None of above." not in data["options"]:
                for idxs in data["sampled_idxs"]:
                    idxs.append(len(data["options"]))
                data["options"].append("None of above.")
        self.dataset.instruction = "none_of_above"
        super().generate_responses()

    def abstain(self):
        self.generate_responses()
        self.parse_freeform_responses()
        for data in self.dataset.get_split(self.split_flag):
            data["abstain"] = [
                r == len(data["options"]) - 1 for r in data["extracted_responses"]
            ]
        self.save_response()
