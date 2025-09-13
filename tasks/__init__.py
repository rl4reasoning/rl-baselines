from typing import Any, Dict

from transformers import AutoTokenizer


def get_dataset(task: str):
    if task == "countdown":
        from .countdown import get_dataset

        return get_dataset()
    elif task.__contains__("math") or task.__contains__("deepscaler") or task.__contains__("chase"):
        from .math import get_dataset

        return get_dataset(task)

    elif task.__contains__("graph"):
        from .graph import get_dataset

        return get_dataset(task)
    else:
        raise ValueError(f"Task {task} not found in get_dataset")


def preprocess_example(
    example: Dict[str, Any],
    task: str,
    tokenizer: AutoTokenizer,
    model_name: str = None,
):
    if task == "countdown":
        from .countdown import preprocess_example

        return preprocess_example(example, tokenizer, model_name)
    elif task.__contains__("math") or task.__contains__("deepscaler") or task.__contains__("chase"):
        from .math import preprocess_example

        return preprocess_example(example, tokenizer, model_name)

    elif task.__contains__("graph"):
        from .graph import preprocess_example

        return preprocess_example(example, tokenizer, model_name)

    else:
        raise ValueError(f"Task {task} not found in preprocess_example")


def get_reward_fn(task: str):
    if task == "countdown":
        from .countdown import compute_reward

        return compute_reward
    elif task.__contains__("math") or task.__contains__("deepscaler") or task.__contains__("chase"):
        from .math import compute_reward

        return compute_reward
    elif task.__contains__("graph"):
        from .graph import compute_reward

        return compute_reward
    else:
        raise ValueError(f"Task {task} not found in get_reward_fn")


def get_extract_answer_fn(task: str):
    if task.__contains__("graph"):
        from .graph import extract_answer

        return extract_answer
    else:
        raise ValueError(f"Task {task} not found in get_extract_answer_fn")


def check_prover_dataset_validity(task: str, prover_train_dataset, prover_tokenizer, train_dataset, tokenizer):
    if task.__contains__("graph"):
        from .graph import check_prover_dataset_validity

        return check_prover_dataset_validity(prover_train_dataset, prover_tokenizer, train_dataset, tokenizer)

    else:
        raise NotImplementedError(f"Task {task} does not have a prover")
