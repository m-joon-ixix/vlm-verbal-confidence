import numpy as np

from method.base import BaseMethod
from method.utils import clean_extracted_answers
from common.const import IDX_TO_LETTER, LETTER_TO_IDX, ABSTAIN_TOKEN, FAILED_TOKEN
from common.model_names import DType
from generate.common import batch_query_llm


class CalibrationAnalysis(BaseMethod):
    method_name = "analysis"

    def generate_confidence(self):
        # instruction = '### Instruction:\nLook at the image and estimate how confident you can understand the image correctly. Provide the confidence in decimal number between 0.0 and 1.0. Give ONLY the probability, no other words or explanation.\n'
        # query_list = self.dataset.customize_queries(
        #     key=f'confidence_image_before',
        #     dtype='gpt',
        #     split_flag=self.split_flag,
        #     sample_size=self.sample_size,
        #     instruction_func=lambda data, idx: instruction,
        #     include_question=False,
        #     include_option=False
        # )
        # _, responses = batch_query_llm(
        #     query_list,
        #     model_name='gpt-4o',
        #     temperature=0.0,
        # )
        # for data in self.dataset.get_split(self.split_flag):
        #     processed_confidence = []
        #     for idx in data['confidence_image_before']:
        #         try:
        #             processed_confidence.append(float(responses[idx]))
        #         except:
        #             processed_confidence.append(FAILED_TOKEN)
        #     data['confidence_image_before'] = processed_confidence
        # self.save_response()
        #
        # instruction = '### Image description: \n{}\n### Instruction:\nProvide the probability that the image description correctly depicts the image in decimal number between 0.0 and 1.0. Give ONLY the probability, no other words or explanation.\n'
        # query_list = self.dataset.customize_queries(
        #     key=f'confidence_image_after',
        #     dtype='gpt',
        #     split_flag=self.split_flag,
        #     sample_size=self.sample_size,
        #     instruction_func=lambda data, idx: instruction.format(data['description']),
        #     include_question=False,
        #     include_option=False
        # )
        # _, responses = batch_query_llm(
        #     query_list,
        #     model_name='gpt-4o',
        #     temperature=0.0,
        # )
        # for data in self.dataset.get_split(self.split_flag):
        #     processed_confidence = []
        #     for idx in data['confidence_image_after']:
        #         try:
        #             processed_confidence.append(float(responses[idx]))
        #         except:
        #             processed_confidence.append(FAILED_TOKEN)
        #     data['confidence_image_after'] = processed_confidence
        # self.save_response()
        #
        # instruction = '### Instruction:\nRead the question and estimate how confident you can answer the question correctly. Provide the confidence in decimal number between 0.0 and 1.0. Give ONLY the probability, no other words or explanation.\n'
        # query_list = self.dataset.customize_queries(
        #     key=f'confidence_all_before',
        #     dtype='gpt',
        #     split_flag=self.split_flag,
        #     sample_size=self.sample_size,
        #     instruction_func=lambda data, idx: instruction,
        # )
        # _, responses = batch_query_llm(
        #     query_list,
        #     model_name='gpt-4o',
        #     temperature=0.0,
        # )
        # for data in self.dataset.get_split(self.split_flag):
        #     processed_confidence = []
        #     for idx in data['confidence_all_before']:
        #         try:
        #             processed_confidence.append(float(responses[idx]))
        #         except:
        #             processed_confidence.append(FAILED_TOKEN)
        #     data['confidence_all_before'] = processed_confidence
        # self.save_response()
        #
        # with open('./instruction/ask_for_calibration.txt', encoding='utf-8') as f:
        #     instruction = ''.join(f.readlines())
        # query_list = self.dataset.customize_queries(
        #     key='confidence_all_after',
        #     dtype='gpt',
        #     split_flag=self.split_flag,
        #     sample_size=self.sample_size,
        #     instruction_func=lambda data, i: instruction.format(data['responses'][i])
        # )
        # _, responses = batch_query_llm(
        #     query_list,
        #     model_name=self.model_name,
        #     temperature=0.0,
        # )
        #
        # for data in self.dataset.get_split(self.split_flag):
        #     processed_confidence = []
        #     for idx in data['confidence_all_after']:
        #         try:
        #             processed_confidence.append(float(responses[idx]))
        #         except:
        #             processed_confidence.append(FAILED_TOKEN)
        #     data['confidence_all_after'] = processed_confidence
        # self.save_response()

        instruction = "### Image description:\n{}\n### Instruction:\nRead the question and estimate how confident you can answer the question correctly. Provide the confidence in decimal number between 0.0 and 1.0. Give ONLY the probability, no other words or explanation.\n"
        query_list = self.dataset.customize_queries(
            key=f"confidence_all2_before",
            dtype=DType.OPENAI,
            split_flag=self.split_flag,
            sample_size=self.sample_size,
            include_image=False,
            instruction_func=lambda data, idx: instruction.format(data["description"]),
        )
        _, responses = batch_query_llm(
            query_list,
            model_name="gpt-4o",
            temperature=0.0,
        )
        for data in self.dataset.get_split(self.split_flag):
            processed_confidence = []
            for idx in data["confidence_all2_before"]:
                try:
                    processed_confidence.append(float(responses[idx]))
                except:
                    processed_confidence.append(FAILED_TOKEN)
            data["confidence_all2_before"] = processed_confidence
        self.save_response()

        instruction = "### Image description:\n{}### Proposed answer:\n{}\n### Instruction:\nProvide the probability that the proposed answer is correct in decimal number between 0.0 and 1.0. Give ONLY the probability, no other words or explanation."
        query_list = self.dataset.customize_queries(
            key="confidence_all2_after",
            dtype=DType.OPENAI,
            split_flag=self.split_flag,
            sample_size=self.sample_size,
            include_image=False,
            instruction_func=lambda data, i: instruction.format(
                data["description"], data["description_responses"][i]
            ),
        )
        _, responses = batch_query_llm(
            query_list,
            model_name=self.model_name,
            temperature=0.0,
        )

        for data in self.dataset.get_split(self.split_flag):
            processed_confidence = []
            for idx in data["confidence_all2_after"]:
                try:
                    processed_confidence.append(float(responses[idx]))
                except:
                    processed_confidence.append(FAILED_TOKEN)
            data["confidence_all2_after"] = processed_confidence
        self.save_response()

    def generate_grounding(self):
        with open(f"instruction/generate_description.txt", encoding="utf-8") as f:
            instruction = "".join(f.readlines())
        query_list = self.dataset.customize_queries(
            key=f"description",
            dtype=DType.OPENAI,
            split_flag=self.split_flag,
            sample_size=1,
            instruction_func=lambda data, idx: instruction,
            include_option=False,
        )
        _, responses = batch_query_llm(
            query_list,
            model_name="gpt-4o",
            temperature=0.0,
        )
        for data in self.dataset.get_split(self.split_flag):
            data["description"] = responses[data["description"][0]]
        self.save_response()

    def generate_grounding_based_response(self):
        # instruction = '###Image description:\n{}\n### Instruction:\nChoose one option that best answers the question and explain your reasoning step by step. Keep your answer concise.'
        # query_list = self.dataset.customize_queries(
        #     key=f'description_responses',
        #     dtype='gpt',
        #     split_flag=self.split_flag,
        #     sample_size=self.sample_size,
        #     instruction_func=lambda data, idx: instruction.format(data['description']),
        #     include_image=False,
        # )
        # probs, responses = batch_query_llm(
        #     query_list,
        #     model_name=self.model_name,
        #     temperature=0.0,
        # )
        # for i, data in enumerate(self.dataset.get_split(self.split_flag)):
        #     data['description_responses'] = responses[i * self.sample_size: (i + 1) * self.sample_size]
        # self.parse_freeform_responses(input_key='description_responses', output_key='description_extracted_responses')

        a, b, c, d = 0, 0, 0, 0
        a_bc, a_ac, b_bc, b_ac, c_bc, c_ac, d_bc, d_ac = [], [], [], [], [], [], [], []
        for data in self.dataset.get_split(self.split_flag):
            if (
                data["description_extracted_responses"][0] == data["correct_answer_idx"]
                and data["extracted_responses"][0] == data["correct_answer_idx"]
            ):
                a += 1
                a_bc.append(data["confidence_all2_before"][0])
                a_ac.append(data["confidence_all2_after"][0])
            elif (
                data["description_extracted_responses"][0] != data["correct_answer_idx"]
                and data["extracted_responses"][0] == data["correct_answer_idx"]
            ):
                b += 1
                b_bc.append(data["confidence_all2_before"][0])
                b_ac.append(data["confidence_all2_after"][0])
            elif (
                data["description_extracted_responses"][0] == data["correct_answer_idx"]
                and data["extracted_responses"][0] != data["correct_answer_idx"]
            ):
                c += 1
                c_bc.append(data["confidence_all2_before"][0])
                c_ac.append(data["confidence_all2_after"][0])
            else:
                d += 1
                d_bc.append(data["confidence_all2_before"][0])
                d_ac.append(data["confidence_all2_after"][0])
        print(a, b, c, d)
        print(
            np.mean(a_bc),
            np.mean(a_ac),
            np.mean(b_bc),
            np.mean(b_ac),
            np.mean(c_bc),
            np.mean(c_ac),
            np.mean(d_bc),
            np.mean(d_ac),
        )

    def abstain(self):
        instruction = '### Image description:\n{}\n### Proposed answer:\n{}\n### Instruction:\nThe correct answer is {}. Read the image description and a proposed answer based on the image description. Is the proposed answer wrong because A. the image description is inaccurate or miss key information; B. the reasoning steps are hallucinated? Think step by step and conclude your judgement with "Conclusion: A or B." in a new line.\n'
        query_list = self.dataset.customize_queries(
            key=f"judgement_responses",
            dtype=DType.OPENAI,
            split_flag=self.split_flag,
            sample_size=self.sample_size,
            instruction_func=lambda data, idx: instruction.format(
                data["description"],
                data["description_responses"][idx],
                IDX_TO_LETTER[data["correct_answer_idx"]],
            ),
            skip_func=lambda data, idx: data["description_extracted_responses"][idx]
            == data["correct_answer_idx"],
        )
        _, responses = batch_query_llm(
            query_list,
            model_name="gpt-4o",
            temperature=0.0,
        )
        for data in self.dataset.get_split(self.split_flag):
            data["judgement_responses"] = [
                responses[idx] if idx != FAILED_TOKEN else ABSTAIN_TOKEN
                for idx in data["judgement_responses"]
            ]
        clean_extracted_answers(
            dataset=self.dataset.get_split(self.split_flag),
            pattern=r"Conclusion:(?: |\n)(A|B)",
            post_process=lambda x: LETTER_TO_IDX[x],
            key="judgement_responses",
            new_key="judgement",
        )
        self.save_response()
        a, b, c, d, e = 0, 0, 0, 0, 0
        for data in self.dataset.get_split(self.split_flag):
            if data["description_extracted_responses"][0] == data["correct_answer_idx"]:
                continue
            if (
                data["extracted_responses"][0] == data["correct_answer_idx"]
                and data["judgement"][0] == 0
            ):
                a += 1
            elif (
                data["extracted_responses"][0] == data["correct_answer_idx"]
                and data["judgement"][0] == 1
            ):
                b += 1
            elif (
                data["extracted_responses"][0] != data["correct_answer_idx"]
                and data["judgement"][0] == 0
            ):
                c += 1
            elif (
                data["extracted_responses"][0] != data["correct_answer_idx"]
                and data["judgement"][0] == 1
            ):
                d += 1
            else:
                e += 1
        print(a, b, c, d, e)
