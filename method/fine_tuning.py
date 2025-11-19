from method.base import BaseMethod
from generate.common import batch_query_llm
from common.model_names import model_name_to_dtype


# A class for baselines that use fine-tuning
class FineTuning(BaseMethod):
    method_name = "ft"

    def __init__(self, **kwargs):
        assert kwargs.get("peft_dir") is not None
        super().__init__(**kwargs)

    @property
    def output_name(self):
        return self.method_name + "_" + self.peft_dir.split("/")[-1] + "_" + self.split

    def generate_responses(self):
        dtype = model_name_to_dtype(self.model_name)
        query_list, _ = self.dataset.form_queries(
            dtype, self.split_flag, self.sample_size
        )

        _, responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            peft_dir=self.peft_dir,
            temperature=0.0,
        )

        for i, data in enumerate(self.dataset.get_split(self.split_flag)):
            data["responses"] = responses[
                i * self.sample_size : (i + 1) * self.sample_size
            ]

        self.save_response()
