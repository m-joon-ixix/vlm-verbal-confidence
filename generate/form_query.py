import base64
import random
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from typing import List

from common.const import IDX_TO_LETTER, FAILED_TOKEN, MMMU_IMAGE_TOKEN
from common.model_names import DType, OpenSrcModel
from common.random_utils import get_seed


def _encode_image(image: Image.Image, dtype="png"):
    buffered = BytesIO()
    image.save(buffered, format=dtype)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _decode_image(image_str: str):
    return Image.open(BytesIO(base64.b64decode(image_str)))


def form_mm_query(text, images, dtype=DType.OPENAI):
    if dtype == DType.CLAUDE:
        message = _form_mm_query_claude(text, images)
    elif dtype == DType.OPENAI:
        message = _form_mm_query_openai(text, images)
    elif dtype == DType.GEMINI:
        message = _form_mm_query_gemini(text, images)
    else:
        message = {"role": "user", "content": [{"type": "text", "text": text}]}
        for image in images:
            encoded_image = _encode_image(image)
            message["content"].append(
                {"type": "image", "image": f"data:image/png;base64,{encoded_image}"}
            )

    return [message]


def _form_mm_query_claude(text, images):
    message = {"role": "user", "content": [{"type": "text", "text": text}]}

    for image in images:
        encoded_image = _encode_image(image)
        if len(encoded_image) > 5 * 1024 * 1024:
            print(f"Image {image} size exceeds 5 MB, skip this image.")
            continue

        message["content"].append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": encoded_image,
                },
            }
        )

    return message


def _form_mm_query_openai(text, images):
    message = {"role": "user", "content": [{"type": "text", "text": text}]}

    for image in images:
        message["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_image(image)}"},
            }
        )

    return message


def _form_mm_query_gemini(text, images):
    message = {"role": "user", "parts": [{"text": text}]}

    for image in images:
        encoded_image = _encode_image(image)
        message["parts"].append(
            {"inlineData": {"mimeType": "image/png", "data": encoded_image}}
        )

    return message


def form_multichoice_queries(
    data_list, sample_size, dtype: DType, instruction_name="multi_choice_query"
):
    query_list = []
    sampled_idxs = []
    with open(f"./instruction/{instruction_name}.txt", encoding="utf-8") as f:
        instruction = "".join(f.readlines())
    random.seed(get_seed())
    for data in tqdm(data_list, desc="Forming Multichoice Queries..."):
        for j in range(sample_size):
            query = "### Question:\n" + data["question"] + "\n### Options:\n"
            if "sampled_idxs" in data:
                sampled_idx = data["sampled_idxs"][j]  # use the existing idxs
                # sampled_idx = list(range(len(data["options"])))  # no shuffle
            else:
                sampled_idx = random.sample(
                    range(len(data["options"])), len(data["options"])
                )
            for i in range(len(data["options"])):
                query += f"{IDX_TO_LETTER[i]}. {data['options'][sampled_idx[i]]}\n"
            query += instruction
            query_list.append(form_mm_query(query, [data["image"]], dtype=dtype))
            sampled_idxs.append(sampled_idx)
    return query_list, sampled_idxs


def customize_multichoice_query(
    data_list,
    sample_size,
    key,
    dtype: DType,
    skip_func=lambda x, idx: False,
    instruction_func=lambda x, idx: "",
    include_question=True,
    include_option=True,
    include_image=True,
):
    query_list = []
    idx = 0
    for data in tqdm(data_list, desc="Customizing Multichoice Queries..."):
        data[key] = []
        for i in range(sample_size):
            if skip_func(data, i):
                data[key].append(FAILED_TOKEN)
                continue
            query = ""
            if include_question:
                query += "### Question:\n" + data["question"] + "\n"
            if include_option:
                query += "### Options:\n"
                for j in range(len(data["options"])):
                    query += f"{IDX_TO_LETTER[j]}. {data['options'][data['sampled_idxs'][i][j]]}\n"
            query += instruction_func(data, i)
            query_list.append(
                form_mm_query(
                    query, [data["image"]] if include_image else [], dtype=dtype
                )
            )
            data[key].append(idx)
            idx += 1
    return query_list


def form_open_ended_queries(
    data_list, sample_size, dtype: DType, instruction_name="open_ended_query"
):
    with open(f"./instruction/{instruction_name}.txt", encoding="utf-8") as f:
        instruction = "".join(f.readlines())

    query_list = []
    for data in tqdm(data_list, desc="Forming Open-Ended Queries..."):
        for _ in range(sample_size):
            query = "### Question:\n" + data["question"] + "\n"
            query += instruction
            query_list.append(form_mm_query(query, [data["image"]], dtype=dtype))

    return query_list


def customize_open_ended_query(
    data_list,
    sample_size,
    dtype: DType,
    instruction_func=lambda x, idx: "",
    include_question=True,
    include_image=True,
):
    query_list = []
    for data in tqdm(data_list, desc="Customizing Open-Ended Queries..."):
        for i in range(sample_size):
            query = ""
            if include_question:
                query += "### Question:\n" + data["question"] + "\n"
            query += instruction_func(data, i)
            query_list.append(
                form_mm_query(
                    query, [data["image"]] if include_image else [], dtype=dtype
                )
            )

    return query_list


def form_multi_turn_queries_for_finetuned_model(
    data_list: List[dict],
    dtype: DType,
    sample_size: int,
    image_desc_attr: str,
    target_attr: str,
) -> List[list]:
    """
    Form 3-turn queries that are used to query the answer to a 2-stage finetuned model

    Args:
        image_desc_attr: attribute containing the image desc that is already generated
        target_attr: attribute that the generated response will be wrote into
    Return:
        A list of queries. Each query contains 3 messages (3-turn conversation)
        Each conversation: user querying for the image desc => assistant answering the image desc => user querying for the answer
    """
    with open("instruction/generate_description.txt", encoding="utf-8") as f:
        description_instruction = "".join(f.readlines())

    # 1. user querying for the image desc
    image_desc_query_list = customize_multichoice_query(
        data_list,
        sample_size=1,
        key=target_attr,
        dtype=dtype,
        instruction_func=lambda data, idx: description_instruction,
        include_option=False,
    )

    # 2. assistant answering the image desc
    image_desc_response_list = [
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": data[image_desc_attr]}],
            }
        ]
        for data in data_list
    ]

    # 3. user querying for the answer
    answer_query_list, _ = form_multichoice_queries(data_list, sample_size, dtype)

    query_list = []
    for i in range(len(data_list)):
        for j in range(sample_size):
            three_turn_query = []
            three_turn_query.extend(image_desc_query_list[i])
            three_turn_query.extend(image_desc_response_list[i])
            # NOTE: `answer_query_list` is `sample_size` times bigger than other lists
            three_turn_query.extend(answer_query_list[i * sample_size + j])
            query_list.append(three_turn_query)

    return query_list


def replace_mmmu_image_token(text: str) -> str:
    return text.replace(MMMU_IMAGE_TOKEN, "<image>")


def build_finetune_user_content(text: str, model_name: str) -> list:
    assert model_name in [_model.value for _model in OpenSrcModel]

    if MMMU_IMAGE_TOKEN in text:
        if model_name == OpenSrcModel.LLAVA.value:
            # NOTE: LLaVA's chat template sends the image portion to the front, irrespective of where the image portion was. Thus, we simply manually replace it to LLaVA's image token ("<image>")
            return [{"type": "text", "text": replace_mmmu_image_token(text)}]
        else:
            content = []
            for split_text in text.split(MMMU_IMAGE_TOKEN):
                content.append({"type": "text", "text": split_text})
                # this replaces the image token that was originally in the text
                content.append({"type": "image"})
            return content[:-1]  # return except the last image portion
    else:
        # if there is no image token in text, just attach the image at the end
        return [{"type": "text", "text": text}, {"type": "image"}]


def build_finetune_assistant_content(text: str) -> list:
    return [{"type": "text", "text": replace_mmmu_image_token(text)}]
