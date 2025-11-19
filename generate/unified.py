import random
import time
import collections
from typing import List
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import openai
import anthropic
from google import genai
from google.genai.types import GenerateContentConfig
from google.genai.errors import APIError

from common.const import FAILED_TOKEN, PROHIBITED_CONTENT_MESSAGE
from common.yaml_utils import load_secret
from common.model_names import DType, model_name_to_dtype
from common.random_utils import get_seed

RETRY_INTERVAL = 3


def query_unified(prompt, index, model, max_tokens, temperature, logprobs):
    if "claude" in model.lower():
        return _query_unified_claude(
            prompt, index, model, max_tokens, temperature, logprobs
        )
    elif "gemini" in model.lower():
        return _query_unified_gemini(
            prompt, index, model, max_tokens, temperature, logprobs
        )
    else:
        return _query_unified_openai(
            prompt, index, model, max_tokens, temperature, logprobs
        )


def _query_unified_claude(prompt, index, model, max_tokens, temperature, logprobs):
    api_keys = _get_api_keys("claude")
    clients = [anthropic.Anthropic(api_key=api_key) for api_key in api_keys]
    # TODO (if needed): implement client & API key rotation such as Gemini
    client = clients[0]

    retry_count = 2
    for _ in range(retry_count):
        try:
            # can't put in a fixed seed value to Anthropic models
            response = client.messages.create(
                messages=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )
            msg = response.content[0].text
            if msg is None or not msg.strip():
                print("Message is empty, retrying...")
                continue

            log_prob = 0.0
            if logprobs:
                log_prob = []
                for prob in response.choices[0].logprobs.content:
                    if -prob.logprob != 9999.0:
                        log_prob.append(-prob.logprob)
                log_prob = sum(log_prob) / len(log_prob)
            return index, log_prob, msg
        except Exception as e:
            print("Retrying due to Error: ", e)
            time.sleep(RETRY_INTERVAL)

    return index, 0.0, FAILED_TOKEN


def _query_unified_gemini(prompt, index, model, max_tokens, temperature, logprobs):
    api_keys = _get_api_keys("gemini")
    # rotate multiple API keys to avoid the rate limit error
    clients = [genai.Client(api_key=api_key) for api_key in api_keys]
    client_idx = random.randint(0, len(clients) - 1)

    retry_count = 10
    for _ in range(retry_count):
        try:
            client = clients[client_idx]
            config = GenerateContentConfig(
                temperature=temperature,
                top_p=1.0,
                # candidate_count=1,
                max_output_tokens=max_tokens,
                seed=get_seed(),
            )
            response = client.models.generate_content(
                contents=prompt,
                model=model,
                config=config,
            )
            msg = response.text
            if msg is None or len(msg.strip()) == 0:
                finish_reason = _get_gemini_finish_reason(response)
                if finish_reason == "MAX_TOKENS":
                    max_tokens *= 2
                if finish_reason == "PROHIBITED_CONTENT":
                    return index, 0.0, PROHIBITED_CONTENT_MESSAGE

                print(f"Message empty due to: {finish_reason} => retrying...")
                continue

            log_prob = 0.0
            if logprobs:
                log_prob = []
                for prob in response.choices[0].logprobs.content:
                    if -prob.logprob != 9999.0:
                        log_prob.append(-prob.logprob)
                log_prob = sum(log_prob) / len(log_prob)

            time.sleep(3)  # sleep to not exceed rate limit
            return index, log_prob, msg
        except Exception as e:
            if isinstance(e, APIError) and e.code == 429:
                print(f"[Client {client_idx}]", end=" ")
                _handle_gemini_429_error(e)
            else:
                print(f"[Client {client_idx}] Retrying due to Error: ", e)
                time.sleep(RETRY_INTERVAL)
        finally:
            client_idx = (client_idx + 1) % len(clients)

    return index, 0.0, FAILED_TOKEN


def _query_unified_openai(prompt, index, model, max_tokens, temperature, logprobs):
    api_keys = _get_api_keys("openai")
    clients = [openai.OpenAI(api_key=api_key) for api_key in api_keys]

    retry_count = 5
    for _ in range(retry_count):
        try:
            # TODO (if needed): implement client & API key rotation such as Gemini
            client = clients[0]
            params = {"messages": prompt, "model": model}
            if any([exp in model.lower() for exp in ["o1", "o3", "o4"]]):
                params["max_completion_tokens"] = max_tokens
                # they use reasoning tokens => do not support params such as temperature, seed, etc.
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature
                params["top_p"] = 1.0
                params["logprobs"] = logprobs
                params["seed"] = get_seed()
            response = client.chat.completions.create(**params)
            msg = response.choices[0].message.content
            if msg is None or len(msg.strip()) == 0:
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    max_tokens *= 2

                print(f"Message empty due to: {finish_reason} => retrying...")
                continue

            log_prob = 0.0
            if logprobs:
                log_prob = []
                for prob in response.choices[0].logprobs.content:
                    if -prob.logprob != 9999.0:
                        log_prob.append(-prob.logprob)
                log_prob = sum(log_prob) / len(log_prob)
            return index, log_prob, msg

        except Exception as e:
            print("Retrying due to Error: ", e)
            time.sleep(RETRY_INTERVAL)

    return index, 0.0, FAILED_TOKEN


# def batch_query_openai(prompt_list, model_name='gpt-4o-mini', max_new_tokens=768, temperature=0.0, output_logits=False):
#     with ProcessPoolExecutor(max_workers=8) as executor:
#         futures = [executor.submit(query_openai, prompt, index, model_name, max_new_tokens, temperature, output_logits) for index, prompt in
#                    enumerate(prompt_list)]
#         response_dict = collections.defaultdict(str)
#         log_prob_dict = collections.defaultdict(str)
#         for job in tqdm(as_completed(futures), total=len(futures), desc="querying openai..."):
#             index, log_prob, res = job.result(timeout=None)
#             response_dict[index] = res
#             log_prob_dict[index] = log_prob

#     return [log_prob_dict[i] for i in range(len(prompt_list))], [response_dict[i] for i in range(len(prompt_list))]


def batch_query_unified(
    prompt_list,
    model_name="gpt-4o",
    max_new_tokens=1536,
    temperature=0.0,
    output_logits=False,
):
    print_prompt_example(prompt_list[0], model_name)

    # Not sure why the API is called in single-threaded environment. TODO: remove if not used
    if "o1" in model_name or "gemini-2.0" in model_name:
        # single thread
        response_dict = collections.defaultdict(str)
        log_prob_dict = collections.defaultdict(str)

        for index, prompt in tqdm(
            enumerate(prompt_list),
            total=len(prompt_list),
            desc=f"querying {model_name} API (single-thread)...",
        ):
            index, log_prob, res = query_unified(
                prompt, index, model_name, max_new_tokens, temperature, output_logits
            )
            response_dict[index] = res
            log_prob_dict[index] = log_prob
    else:
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    query_unified,
                    prompt,
                    index,
                    model_name,
                    max_new_tokens,
                    temperature,
                    output_logits,
                )
                for index, prompt in enumerate(prompt_list)
            ]
            response_dict = collections.defaultdict(str)
            log_prob_dict = collections.defaultdict(str)
            for job in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"querying {model_name} API...",
            ):
                index, log_prob, res = job.result(timeout=None)
                response_dict[index] = res
                log_prob_dict[index] = log_prob

    return [log_prob_dict[i] for i in range(len(prompt_list))], [
        response_dict[i] for i in range(len(prompt_list))
    ]


def print_prompt_example(prompt, model_name):
    print(f"An example of prompt (image part is not printed):")
    print("=" * 100)

    dtype = model_name_to_dtype(model_name)
    # check the `if`, `elif` statements in the method `form_mm_query()` for each case
    if dtype == DType.OPENAI:
        # _form_mm_query_openai()
        print(prompt[0]["content"][0]["text"])
    elif dtype == DType.CLAUDE:
        # _form_mm_query_claude()
        print(prompt[0]["content"][0]["text"])
    elif dtype == DType.GEMINI:
        # _form_mm_query_gemini()
        print(prompt[0]["parts"][0]["text"])
    else:
        print(
            f"Warning from print_prompt_example() - Unexpected model name: {model_name}"
        )

    print("=" * 100)


def _get_gemini_finish_reason(response) -> str:
    try:
        if response.candidates:
            return response.candidates[0].finish_reason.value
        elif response.prompt_feedback:
            return response.prompt_feedback.block_reason.value
        else:
            return str(response)
    except Exception:
        return str(response)


def _handle_gemini_429_error(e: Exception):
    # HTTP 429: Too Many Requests => Exceeded Quota
    try:
        debug_info = {
            k: v
            for k, v in e.details["error"]["details"][0]["violations"][0].items()
            if k in ["quotaId", "quotaValue"]
        }
        retry_delay = e.details["error"]["details"][-1]["retryDelay"]
        debug_info["retryDelay"] = retry_delay
        print(f"Retrying due to Gemini 429 Error: {debug_info}")
    except Exception:  # if error occurred while parsing
        print(f"Retrying due to Gemini 429 Error: ", e)

    # time.sleep(int(retry_delay.replace("s", "")) + 1)
    # NOTE: just sleep for a fixed time since multiple API keys are being rotated
    time.sleep(RETRY_INTERVAL)


def _get_api_keys(model_family: str) -> List[str]:
    assert model_family in ["openai", "gemini", "claude"]

    api_keys = load_secret("api_key")[model_family]
    if len(api_keys) == 0:
        raise AssertionError(f"No API keys found for {model_family}")

    return api_keys
