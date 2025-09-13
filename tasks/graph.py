from typing import Any, Dict, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

from .templates import templates_dict

GRAPH_PROBLEM_TEMPLATE = (
    "Given a bi-directional graph in the form of space separated edges, output a path from source node "
    "to the destination node in the form of comma separated integers.\n"
    "For this question the graph is {graph}\n"
    "The source node is {source}\n"
    "The destination node is {destination}\n"
)


def get_dataset(task: str):
    # Try loading dataset from huggingface first
    dataset = load_dataset(task)
    train_dataset = dataset["train"].shuffle(seed=42)
    test_dataset = dataset["test"].shuffle(seed=42).select(range(50))

    return {"train": train_dataset, "test": {"graph-test": test_dataset}}


def get_system_user_prompt(model_name: str):
    if model_name.lower().__contains__("r1"):
        return templates_dict["r1"]["system_prompt"], templates_dict["r1"]["prompt_template"], templates_dict["r1"]["apply_tokenizer_template"]
    else:
        return (
            templates_dict["qwen_math"]["system_prompt"],
            templates_dict["qwen_math"]["prompt_template"],
            templates_dict["qwen_math"]["apply_tokenizer_template"],
        )


def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    model_name: str = None,
):
    graph = example["graph"]
    # Replace all '|' with ' ' -- model works better with spaces
    graph = graph.replace("|", " ")
    source = example["source"]
    destination = example["destination"]
    _, prompt_template, apply_tokenizer_template = get_system_user_prompt(model_name)
    prefix = []
    problem = GRAPH_PROBLEM_TEMPLATE.format(graph=graph, source=source, destination=destination)
    if apply_tokenizer_template:
        prefix = [{"role": "user", "content": prompt_template.format(problem)}]
        input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True, continue_final_message=False)
        prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    else:
        prompt = prompt_template.format(problem)
        input_ids = tokenizer.encode(prompt)
    return {"prompt": prompt, "input_ids": input_ids}


def check_prover_dataset_validity(prover_train_dataset, prover_tokenizer, train_dataset, tokenizer):
    assert len(prover_train_dataset) == len(train_dataset), "Prover train dataset and train dataset should have the same length"
    assert tokenizer.eos_token_id == prover_tokenizer.eos_token_id, "Tokenizer and prover tokenizer should have the same EOS token id"
    assert tokenizer.pad_token_id == prover_tokenizer.pad_token_id, "Tokenizer and prover tokenizer should have the same PAD token id"
    for i in range(len(prover_train_dataset)):
        prover_sample_token_ids = prover_train_dataset[i]["input_ids"]
        train_sample_token_ids = train_dataset[i]["input_ids"]

        len_train_sample_token_ids = len(train_sample_token_ids)
        len_prover_sample_token_ids = len(prover_sample_token_ids)
        assert (
            len_train_sample_token_ids == len_prover_sample_token_ids
        ), "Prover sample token ids should be same length as the train sample token ids"
        prover_sample_decoded = prover_tokenizer.decode(prover_sample_token_ids)
        train_sample_decoded = tokenizer.decode(train_sample_token_ids)
        assert prover_sample_decoded == train_sample_decoded, "Train tokenizer and prover tokenizer should decode to the same thing"


def extract_answer(answer):
    try:
        extracted_answer = answer.split("\\boxed{")[-1].split("}")[0]
        normalized_extracted_answer = extracted_answer.replace(" ", "")
    except Exception as e:
        print(f"Error in extracting answer: {e}")
        normalized_extracted_answer = "No answer"
        return normalized_extracted_answer
    return normalized_extracted_answer


def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    return 0.0


def answer_reward_func(completion: str, solution: str) -> float:
    normalized_extracted_answer = extract_answer(completion)
    normalized_ground_truth_answer = solution.replace(" ", "")
    if normalized_extracted_answer == normalized_ground_truth_answer:
        return 1.0
    else:
        return 0.0


def compute_reward(completion: str, sample: Dict[str, Any], EOS_TOKEN: str) -> Tuple[float, Dict[str, float]]:
    solution = sample["path"]

    format_reward = format_reward_func(completion, EOS_TOKEN)
    answer_reward = answer_reward_func(completion=completion, solution=solution)

    reward = format_reward + answer_reward

    metrics = {
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }
    return reward, metrics
