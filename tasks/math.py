from typing import Any, Dict, Tuple

from datasets import concatenate_datasets, load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from transformers import AutoTokenizer

from .templates import templates_dict


def get_dataset(task: str):
    if task == "math":
        subsets = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
        subset_datasets = [
            load_dataset("EleutherAI/hendrycks_math", subset, split="train", download_mode="reuse_dataset_if_exists") for subset in subsets
        ]
        train_dataset = concatenate_datasets(subset_datasets)
        # test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    else:
        train_dataset = load_dataset(task, split="train")

    test_datasets = {
        # "math-500": load_dataset("HuggingFaceH4/MATH-500", split="test"),
        "hmmt-feb-2025": load_dataset("MathArena/hmmt_feb_2025", split="train"),
        "aime-2024": load_dataset("HuggingFaceH4/aime_2024", split="train"),
        "aime-2025": load_dataset("MathArena/aime_2025", split="train"),
    }
    return {"train": train_dataset, "test": test_datasets}


def get_system_user_prompt(model_name: str):
    if model_name.lower().__contains__("r1"):
        return templates_dict["r1"]["system_prompt"], templates_dict["r1"]["prompt_template"], templates_dict["r1"]["apply_tokenizer_template"]
    elif model_name.__contains__("Qwen2.5") or model_name.__contains__("Qwen3"):
        return (
            templates_dict["qwen_math"]["system_prompt"],
            templates_dict["qwen_math"]["prompt_template"],
            templates_dict["qwen_math"]["apply_tokenizer_template"],
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")


def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    model_name: str = None,
):
    if "problem" in example:
        problem: str = example["problem"]
    elif "question" in example:  # Added for math squared
        problem: str = example["question"]
    else:
        raise ValueError(f"Problem not found in example: {example}")
    system_prompt, prompt_template, apply_tokenizer_template = get_system_user_prompt(model_name)
    prefix = []
    if apply_tokenizer_template:  # DeepSeek-R1 distil
        # Not using system prompt for DeepSeek-R1 distil:
        # see https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B#usage-recommendations
        prefix = [
            {
                "role": "user",
                "content": prompt_template.format(problem),
            },
        ]
        input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True, continue_final_message=False)
        prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    else:
        # Qwen-Math, R1, Qwen2.5, Qwen3
        prompt = prompt_template.format(problem)
        input_ids = tokenizer.encode(prompt)

    return {"prompt": prompt, "input_ids": input_ids}


def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    return 0.0


def answer_reward_func(completion: str, solution: str, timeout_score: float = 0.0) -> float:
    try:
        # Refer: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py#L45-L68
        extracted_model_answer = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        extracted_ground_truth_answer = parse(
            solution,
            extraction_mode="first_match",
        )
        ret_score = verify(extracted_ground_truth_answer, extracted_model_answer)
    except TimeoutException:
        ret_score = timeout_score
    return ret_score


def compute_reward(completion: str, sample: Dict[str, Any], EOS_TOKEN: str) -> Tuple[float, Dict[str, float]]:
    if "answer" in sample:
        if str(sample["answer"]).__contains__("\\boxed"):
            solution = sample["answer"]
        else:
            boxed_answer = f'\\boxed{{{sample["answer"]}}}'
            solution = boxed_answer
    elif "final_answer" in sample and sample["final_answer"] != "-":  # for math squared
        if str(sample["final_answer"]).__contains__("\\boxed"):
            solution = sample["final_answer"]
        else:
            boxed_answer = f'\\boxed{{{sample["final_answer"]}}}'
            solution = boxed_answer
    else:  # Use the solution only when answer is not present in the dataset
        assert "solution" in sample, "Either solution or answer must be present in the sample"
        solution = sample["solution"]
        if solution == "":
            print("[WARNING] Going to use empty solution")

    format_reward = format_reward_func(completion, EOS_TOKEN)
    answer_reward = answer_reward_func(completion=completion, solution=solution)

    reward = format_reward + answer_reward

    metrics = {
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }

    return reward, metrics
