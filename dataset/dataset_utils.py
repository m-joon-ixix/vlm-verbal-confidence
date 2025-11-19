import os
from tqdm import tqdm
from PIL import Image

from common.json_utils import load_from_json, dump_to_json


def load_custom_dataset(
    dataset_name, sample_size=2000, load_flag="e", load_from_exist=True
):
    if dataset_name == "MMMU":
        from dataset.mmmu import MmmuDataset

        dataset_class = MmmuDataset
    elif "MMMUPro" in dataset_name:
        from dataset.mmmu_pro import MmmuProDataset

        dataset_class = MmmuProDataset
    elif dataset_name == "AOKVQA":
        from dataset.a_okvqa import AOkvqaDataset

        dataset_class = AOkvqaDataset
    elif dataset_name == "MathVista":
        from dataset.math_vista import MathVistaDataset

        dataset_class = MathVistaDataset
    elif dataset_name == "OCRBenchV2":
        from dataset.ocrbench_v2 import OcrBenchV2Dataset

        dataset_class = OcrBenchV2Dataset
    else:
        raise NotImplementedError

    return dataset_class(
        dataset_name=dataset_name,
        sample_size=sample_size,
        load_flag=load_flag,
        load_from_exist=load_from_exist,
    )


def load_json_data_with_image(data_list, load_dir, load_image_dir, load_name):
    """
    `data_list` should be a reference to a List instance
    """
    load_image_name = load_name.split("_")[-1]
    json_data = load_from_json(f"{load_dir}/{load_name}.json")

    # clear the original List instance, and add new data to this instance
    data_list.clear()

    if _get_sample_count():
        json_data = json_data[: _get_sample_count()]

    for sample in json_data:
        data_list.append(sample)

    for data in data_list:
        with Image.open(
            f'{load_image_dir}/{load_image_name}_image/{data["id"]}.png'
        ) as image:
            data["image"] = image.copy()


def _get_sample_count():
    # if env is given, only use the first N examples (this is for test purpose)
    # CAUTION: If the result file already exists, this might overwrite the whole file with only a few examples left
    if os.getenv("DATASET_SAMPLE_COUNT"):
        return int(os.getenv("DATASET_SAMPLE_COUNT"))
    else:
        return None


def save_json_data(data_list, save_dir, save_image_dir, save_name, save_image=False):
    os.makedirs(save_dir, exist_ok=True)

    dump_to_json(
        f"{save_dir}/{save_name}.json",
        [{k: v for k, v in data.items() if k != "image"} for data in data_list],
    )

    if save_image:
        _save_image_dir = f"{save_image_dir}/{save_name}_image"
        os.makedirs(_save_image_dir, exist_ok=True)
        for data in tqdm(data_list, desc="Saving images..."):
            data["image"].save(f"{_save_image_dir}/{data['id']}.png", format="png")
