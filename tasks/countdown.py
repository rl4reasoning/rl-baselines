import re
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

SYSTEM_MESSAGE = "You are a helpful assistant. You first think about the reasoning process in the mind " "and then provide the user with the answer."
PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final equation and answer in "
    "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
)


def get_dataset():
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    revised_dataset = dataset.train_test_split(test_size=500, seed=42)
    return {"train": revised_dataset["train"], "test": {"countdown": revised_dataset["test"]}}



def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    model_name: str = None,
):
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>",
        },
    ]
    input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)
    prompt = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {"prompt": prompt, "input_ids": input_ids}


def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    """
    Format: <think>...</think><answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Synthetically prepend <think> (if your pipeline relies on that to ease matching)
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def answer_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(completion: str, sample: Dict[str, Any], EOS_TOKEN: str) -> Tuple[float, Dict[str, float]]:
    nums = sample["nums"]
    target = sample["target"]

    format_reward = format_reward_func(completion, EOS_TOKEN)
    answer_reward = answer_reward_func(completion=completion, nums=nums, target=target)

    reward = format_reward + answer_reward

    metrics = {
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }

    return reward, metrics
