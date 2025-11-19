from method.r_tuning import RTuning
from metric import compute_metrics_from_dataset

if __name__ == "__main__":
    a = RTuning(
        dataset_name="MMMUPro",
        # model_name='claude-3-5-sonnet-20241022',
        # model_name='gemini-2.0-flash-exp',
        # model_name='chatgpt-4o-latest',
        # model_name='llama',
        model_name="qwen",
        peft_dir="",
        split_flag="t",
        response_sample_size=8,
    )
    print(
        f"Dataset: {a.dataset_name}\n"
        f"Model name: {a.model_name}\n"
        f"Data split: {a.split_flag}"
    )
    a.abstain()
    print(
        compute_metrics_from_dataset(a.dataset.get_split(a.split_flag))
    )  # for evaluation
