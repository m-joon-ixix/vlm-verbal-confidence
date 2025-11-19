from method.base import BaseMethod
from generate.common import batch_query_llm
from common.const import FAILED_TOKEN
from common.model_names import model_name_to_dtype


class SelfReflect(BaseMethod):
    method_name = "reflect"

    def abstain(self):
        assert self.method_name == "reflect" or self.method_name == "reflect_cot"
        if not all(
            ["judgement" in data for data in self.dataset.get_split(self.split_flag)]
        ):
            with open(f"instruction/{self.method_name}.txt", encoding="utf-8") as f:
                reflect_instruction = "".join(f.readlines())
            query_list = self.dataset.customize_queries(
                key="judgement",
                dtype=model_name_to_dtype(self.model_name),
                split_flag=self.split_flag,
                sample_size=self.sample_size,
                instruction_func=lambda data, i: reflect_instruction.format(
                    data["responses"][i]
                ),
            )
            _, responses = batch_query_llm(
                query_list,
                model_name=self.model_name,
                temperature=0.0,
                max_new_tokens=16,
            )
            for data in self.dataset.get_split(self.split_flag):
                data["judgement"] = [responses[idx] for idx in data["judgement"]]
        for data in self.dataset.get_split(self.split_flag):
            if self.method_name == "reflect":
                data["abstain"] = [
                    (
                        False
                        if "yes" in j.lower()
                        else True if "no" in j.lower() else FAILED_TOKEN
                    )
                    for j in data["judgement"]
                ]
            else:
                data["abstain"] = [
                    j.lower().strip().split("\n")[-1] for j in data["judgement"]
                ]
                data["abstain"] = [
                    (
                        False
                        if "correct" in a and "incorrect" not in a
                        else True if "incorrect" in a else FAILED_TOKEN
                    )
                    for a in data["abstain"]
                ]
        self.save_response()
