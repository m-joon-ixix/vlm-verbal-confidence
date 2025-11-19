# Managing Secrets
> This project manages secrets in a separate yaml file, which is not uploaded to GitHub for security purposes. The file is `config/secrets.yaml` and should be in the following format.
```yaml
hf_key: {YOUR_KEY}
slack_webhook_url: {YOUR_KEY}
api_key:
  openai:
    - {YOUR_KEY}
    # more
  gemini:
    - {YOUR_KEY}
    # more
  claude:
    - {YOUR_KEY}
    # more
```

# Guidelines
> The explanations below can be outdated. They will be organized in the future.

1. Setup the environment using `pip install -r requirements.txt`.
2. Generate origin_test.json and origin_train.json using `generate_responses` in `BaseMethod` class and parse the responses using `parse_structured_response`.
3. If you want to test the prompt-based methods, repeat Step 2 in `YourMethod` class and then call `abstain`.
4. If you want to test the finetuning-based methods:
   1. Construct the training dataset using `construct_dataset` in `RTuning` class.
   2. Finetune the model with the training dataset.
   3. Run inference on test split with the fine-tuned VLM to generate responses and then parse them. 
