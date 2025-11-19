from method.calibration import Calibration
from generate.common import batch_query_llm
from common.const import FAILED_TOKEN
from common.model_names import model_name_to_dtype


class VerbalizedConfidence(Calibration):
    method_name = "verbalize"

    def _generate_confidence(self):
        with open("./instruction/ask_for_calibration.txt", encoding="utf-8") as f:
            instruction = "".join(f.readlines())
        query_list = self.dataset.customize_queries(
            key="confidence",
            dtype=model_name_to_dtype(self.model_name),
            split_flag=self.split_flag,
            sample_size=self.sample_size,
            instruction_func=lambda data, i: instruction.format(data["responses"][i]),
        )
        _, responses = batch_query_llm(
            query_list, model_name=self.model_name, temperature=0.0, max_new_tokens=16
        )

        for data in self.dataset.get_split(self.split_flag):
            processed_confidence = []
            for idx in data["confidence"]:
                try:
                    processed_confidence.append(float(responses[idx]))
                except:
                    processed_confidence.append(FAILED_TOKEN)
            data["confidence"] = processed_confidence
