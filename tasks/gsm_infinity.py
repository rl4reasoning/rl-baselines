from typing import Any, Dict, Tuple

from datasets import concatenate_datasets, load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from transformers import AutoTokenizer

def get_dataset(task: str):
    train_dataset = load_dataset(task, split='train')
    test_datasets = {
        f'{task.split("/")[-1]}_test': load_dataset(task, split='test')
    }
    return {'train': train_dataset, 'test': test_datasets}

def preprocess_example(example: Dict[str, Any], tokenizer: AutoTokenizer, model_name: str = None):
    assert 'messages' in example, f"Messages not found in example: {example}"
    input_ids = tokenizer.apply_chat_template(example['messages'], tokenize=True)
    
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    
    return {'prompt': prompt, 'input_ids': input_ids}

def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    return 0.0

def answer_reward_func(completion: str, solution: str, timeout_score: float = 0.0) -> float:
    try:
        extracted_model_answer = parse(completion, extraction_mode="first_match")
        extracted_ground_truth_answer = parse(solution, extraction_mode="first_match")
        ret_score = verify(extracted_ground_truth_answer, extracted_model_answer)
    except TimeoutException:
        ret_score = timeout_score
    return ret_score

def compute_reward(completion: str, sample: Dict[str, Any], EOS_TOKEN: str) -> Tuple[float, Dict[str, float]]:
    solution = sample["solution"]
    
    format_reward = format_reward_func(completion, EOS_TOKEN)
    answer_reward = answer_reward_func(completion=completion, solution=solution)
    
    reward = format_reward + answer_reward
    
    metrics = {
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }
    
    return reward, metrics
