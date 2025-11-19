from common.model_names import model_name_to_dtype
from generate.common import batch_query_llm
from method.r_tuning_peft import RTuningPeft


# The class to use when running models finetuned only with answer prefixes
class RTuningPeftOnlyAnswer(RTuningPeft):
    def generate_responses(self):
        dtype = model_name_to_dtype(self.model_name)
        # NOTE: the prompt only needs to include the question with image & options
        query_list, _ = self.dataset.form_queries(
            dtype, self.split_flag, self.sample_size
        )

        _, responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            peft_dir=self.peft_dir,
            temperature=0.0,
            batch_size=self.sample_size,
        )

        for i, data in enumerate(self.dataset.get_split(self.split_flag)):
            data["responses"] = responses[
                i * self.sample_size : (i + 1) * self.sample_size
            ]

        self.save_response()

    def prepare_validation_data(self):
        self.generate_answer_confidence()
        self.determine_optimal_answer_threshold()

    def abstain(self):
        self.abstain_answer()
        self.abstain_answer_confidence()
        self.abstain_make_decisions()

    def abstain_answer(self):
        query_list, _ = self.dataset.form_queries(
            dtype=model_name_to_dtype(self.model_name),
            split_flag=self.split_flag,
            # NOTE: uses the first of "sampled_idxs". CAUTION: if this is to be changed, look at method.utils.extract_multichoice_answers
            sample_size=1,
        )

        _, responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            peft_dir=self.peft_dir,
            temperature=0.0,
            batch_size=1,
            # output_logits=True
        )

        for i, data in enumerate(self.dataset.get_split(self.split_flag)):
            data["abstain_response"] = responses[i]

        self.parse_structured_response(response_attr="abstain_response")
        # `self.save_response()` included above

    def abstain_make_decisions(self):
        if self.answer_threshold is None:
            self._load_optimal_threshold("answer")

        for data in self.dataset.get_split(self.split_flag):
            data["answer_abstain"] = (
                data["abstain_answer_confidence"] / 10.0
            ) < self.answer_threshold

            # final abstain decision is only based on the answer part
            data["abstain"] = [data["answer_abstain"]]

        self.save_response()
