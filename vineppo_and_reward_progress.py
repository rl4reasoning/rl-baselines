"""
Implementation for VinePPO (https://arxiv.org/abs/2410.01679) and Rewarding Progress (https://arxiv.org/abs/2410.08146)
"""

# ─── Load environment variables early ──────────────────────────────────────────
import os
from collections import defaultdict

import yaml

if os.path.exists("env_vars.yml"):
    with open("env_vars.yml", "r") as f:
        env_vars = yaml.safe_load(f)
    for key, value in env_vars.items():
        os.environ[key] = value

import gc

# ─── Standard library ──────────────────────────────────────────────────────────
import itertools
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist

# ─── Third-party libraries ────────────────────────────────────────────────────
import tyro
from deepspeed import DeepSpeedEngine
from deepspeed.runtime.utils import see_memory_usage
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams

import wandb

# ─── Local application imports ─────────────────────────────────────────────────
from tasks import get_dataset, get_reward_fn, preprocess_example
from utils import (
    clean_up_checkpoints,
    close_to_zero,
    compute_token_entropy,
    compute_token_log_probs,
    compute_token_logits,
    dump_episodes,
    evaluate_on_test_set,
    find_last_checkpoint,
    fix_oov_logits_processor,
    initialize_training_process_group,
    load_model_into_vllm,
    prepare_model_inputs,
    update_model_inputs_with_advantages,
)

os.environ["VLLM_USE_V1"] = "0"

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@dataclass
class GRPOConfig:

    num_processes: int = 4

    per_device_train_batch_size: int = 8
    per_device_ppo_mini_batch_size: Optional[int] = None
    num_ppo_epochs: int = 1
    num_generations: int = 5

    per_device_ppo_micro_batch_size: int = 4

    num_iterations: Optional[int] = 1000

    max_response_tokens: int = 4096
    max_prompt_tokens: int = 1024

    max_eval_tokens: int = 4096
    num_generations_eval: int = 16

    learning_rate: float = 1e-6
    kl_coeff: float = 0.001

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    prover_policy_model_name: str = "Qwen/Qwen3-1.7B"

    task: str = "star-graph-deg-10-path-10-nodes-300"
    output_dir: Optional[str] = "results/"  # required field
    run_id: Optional[str] = "Qwen2.5-1.5B-Instruct-Qwen3-1.7B-Bon-4-Prover-Vppo-K-3-Max-Ent-Split-3-Alpha-0.8"
    load_checkpoint: Optional[bool] = True
    save_checkpoint: Optional[bool] = True

    debug: Optional[bool] = False

    # loss_type: str = 'grpo'
    loss_type: str = "dr_grpo"  # choose b/w ["grpo", "dr_grpo"]
    algorithm: str = "vineppo"  # we use this just to name the experiment folder

    epsilon: Optional[float] = 0.2  # Epsilon value for clipping

    # sampling parameters
    temperature: float = 0.6
    top_p: float = 0.999  # to avoid sampling unused tokens absent from
    top_k: int = -1  # no top k
    vineppo_k: int = 1
    top_k_entropy_tokens: int = 3
    # Note that top_k_entropy_tokens = 3 creates 4 chunks, so use max_vine_ppo_chunks = 4
    # top_k_entropy_tokens = -1 to get 4 equally spaced chunks.
    max_vineppo_chunks: int = -1
    prover_policy_best_of_n: int = 4
    # Final advantages are computed as ORM Advantage + alpha * Progress advantage
    prover_alpha: float = 0.8
    current_policy_as_prover: int = 0  # 0: False, 1: True
    split_and_merge_chunks: int = 0

    def __post_init__(self):

        assert self.output_dir is not None

        if self.per_device_ppo_mini_batch_size is None:
            self.per_device_ppo_mini_batch_size = self.per_device_train_batch_size
        else:
            assert self.per_device_train_batch_size % self.per_device_ppo_mini_batch_size == 0

        assert (self.per_device_ppo_mini_batch_size * self.num_generations) % self.per_device_ppo_micro_batch_size == 0
        assert not (
            self.split_and_merge_chunks > 0 and self.top_k_entropy_tokens > 0
        ), "Splitting a response can only be done using top_k_entropy_tokens or max_vineppo_chunks, not both"
        assert self.prover_alpha > 0, "prover_alpha must be positive"
        assert self.prover_alpha <= 1, "prover_alpha must be less than or equal to 1"
        if self.split_and_merge_chunks > 0:
            assert self.max_vineppo_chunks > 0, "max_vineppo_chunks must be set when split_and_merge_chunks is set"
        if self.current_policy_as_prover == 1:
            assert (
                self.prover_policy_model_name == self.model_name
            ), "prover_policy_model_name must be the same as model_name when current_policy_as_prover is set to 1"


def create_training_episodes_and_model_inputs(
    *,
    samples: List[Dict[str, Any]] = None,
    all_generations: List[List[int]] = None,
    all_finish_reasons: List[str] = None,
    tokenizer: AutoTokenizer = None,
    EOS_TOKEN: str = None,
    GENERATIONS_PER_SAMPLE: int = None,
    compute_reward_fn: Callable = None,
    LOSS_TYPE: str = None,
    metrics_dict: Dict[str, Any] = None,
    device: torch.device = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    # Process responses and calculate rewards
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE)) for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    for sample, group_indices in zip(samples, groups):
        response_token_ids = [all_generations[i] for i in group_indices]
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
        rewards_and_metrics = [compute_reward_fn(resp, sample, EOS_TOKEN) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)

        if LOSS_TYPE == "grpo":
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        elif LOSS_TYPE == "dr_grpo":
            advantages = rewards - rewards.mean()
        else:
            raise ValueError(f"Unknown loss type: {LOSS_TYPE}")

        per_token_advantages = [[adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)]

        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(per_token_advantages)

        response_lengths = np.array([len(ids) for ids in response_token_ids])
        terminated_completions_mask = np.array([fr == "stop" for fr in finish_reasons], dtype=np.bool)
        non_terminated_completions_mask = ~terminated_completions_mask

        # log stuff
        metrics_dict["extra/per_episode_rewards_list"].extend(rewards.tolist())  # used for logging in episode table
        metrics_dict["extra/per_episode_response_lengths_list"].extend(response_lengths.tolist())  # used for logging in episode table

        metrics_dict["rewards"].append(rewards.mean())
        metrics_dict["non_stop_rate"].append(np.mean([fr != "stop" for fr in finish_reasons]))
        metrics_dict["response_lengths"].append(np.mean([len(ids) for ids in response_token_ids]))

        metrics_dict["rewards_std"].append(rewards.std())
        metrics_dict["rewards_is_std_zero"].append(np.isclose(rewards.std(), 0.0))

        metrics_dict["response_lengths/mean_length"].append(response_lengths.mean())
        metrics_dict["response_lengths/min_length"].append(response_lengths.min())
        metrics_dict["response_lengths/max_length"].append(response_lengths.max())

        # log for dr grpo correctness checks
        correct_mask = np.array([rm["answer_reward"] > 0 for rm in reward_metrics], dtype=np.bool)
        incorrect_mask = ~correct_mask
        metrics_dict["extra/mean_correct_response_length"].append(response_lengths[correct_mask].mean() if correct_mask.any() else 0.0)
        metrics_dict["extra/mean_incorrect_response_length"].append(response_lengths[incorrect_mask].mean() if incorrect_mask.any() else 0.0)

        # just logging for sanity checks
        metrics_dict["extra/mean_terminated_length"].append(
            response_lengths[terminated_completions_mask].mean() if terminated_completions_mask.any() else 0.0
        )
        metrics_dict["extra/min_terminated_length"].append(
            response_lengths[terminated_completions_mask].min() if terminated_completions_mask.any() else 0.0
        )
        metrics_dict["extra/max_terminated_length"].append(
            response_lengths[terminated_completions_mask].max() if terminated_completions_mask.any() else 0.0
        )

        metrics_dict["extra/mean_non_terminated_length"].append(
            response_lengths[non_terminated_completions_mask].mean() if non_terminated_completions_mask.any() else 0.0
        )
        metrics_dict["extra/min_non_terminated_length"].append(
            response_lengths[non_terminated_completions_mask].min() if non_terminated_completions_mask.any() else 0.0
        )
        metrics_dict["extra/max_non_terminated_length"].append(
            response_lengths[non_terminated_completions_mask].max() if non_terminated_completions_mask.any() else 0.0
        )

        for rm in reward_metrics:
            for k, v in rm.items():
                metrics_dict[f"reward_metrics/{k}"].append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    # Here both advantages can be combined with model_inputs in one go, we actually don't need
    # two function calls, but in vineppo two function calls will be needed, as we will compute the
    # logprobs from the first model_inputs, then compute entropy at each position, split the trajectory
    # based on max entropy position, and then compute advantages for each sub-trajectory. For the sake of
    # consistency, we keep it as two function calls here also.

    model_inputs = prepare_model_inputs(
        query_token_ids=episodes["all_query_token_ids"],
        response_token_ids=episodes["all_response_token_ids"],
        device=device,
    )

    model_inputs = update_model_inputs_with_advantages(
        query_token_ids=episodes["all_query_token_ids"],
        response_token_ids=episodes["all_response_token_ids"],
        advantages=episodes["all_advantages"],
        model_inputs=model_inputs,
        device=device,
    )

    return episodes, model_inputs


def create_vineppo_training_episodes_and_model_inputs(
    *,
    inference_engine: LLM = None,
    samples: List[Dict[str, Any]] = None,
    all_generations: List[List[int]] = None,
    all_finish_reasons: List[str] = None,
    tokenizer: AutoTokenizer = None,
    EOS_TOKEN_ID: int = None,
    EOS_TOKEN: str = None,
    GENERATIONS_PER_SAMPLE: int = None,
    MAX_RESPONSE_TOKENS: int = None,
    VINEPPO_K: int = None,
    TEMPERATURE: float = None,
    TOP_P: float = None,
    TOP_K: int = None,
    compute_reward_fn: Callable = None,
    metrics_dict: Dict[str, Any] = None,
    device: torch.device = None,
    policy_model: Union[DeepSpeedEngine, PreTrainedModel] = None,
    top_k_entropy_tokens: int = -1,
    max_vineppo_chunks: Optional[int] = None,
    prover_policy_best_of_n: int = 1,
    prover_alpha: float = 1.0,
    LOSS_TYPE: str = None,
    split_and_merge_chunks: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    *** EXPERIMENTAL ***
    Process model generations and calculate rewards for VinePPO training episodes.

    This function implements the VinePPO algorithm,
    which uses Monte Carlo rollouts to estimate state values and compute token-level advantages.
    See: https://arxiv.org/abs/2410.01679

    Note that it only differs from GRPO in the way it computes advantages. So the rest of the code stays the same.

    The algorithm works as follows:
    1. Split each response into intermediate states (every 100 tokens)
    2. For each intermediate state, generate VINEPPO_K Monte Carlo rollouts
    3. Estimate state values by averaging rewards from these rollouts
    4. Compute token-level advantages based on value differences between states

    Args:
        samples: List of input samples, each containing:
            - input_ids: List[int], tokenized input prompt
            - nums: List[int], numbers to use in equation
            - target: int, target value for equation
        all_generations: List of token ID sequences for each generated response
        all_finish_reasons: List of finish reasons for each generation ("stop" or other)

    Returns:
        Tuple containing:
        1. Dictionary with processed data for training:
            - all_query_token_ids: List[List[int]], input token IDs repeated for each generation
            - all_response_token_ids: List[List[int]], response token IDs with EOS tokens added
            - all_advantages: List[List[float]], advantage values repeated for each token
        2. Dictionary with generation statistics:
            - response_lengths: List[int], lengths of generated responses
            - rewards: List[float], raw reward values
            - non_stop_rate: List[bool], whether each generation ended naturally
            - reward_metrics/*: Various reward component metrics
    """

    def split_tokens_at_delimiter(token_ids, delimiter_token_ids):
        chunks, current_chunk = [], []
        for token_id in token_ids:
            current_chunk.append(token_id)
            if token_id in delimiter_token_ids:
                chunks.append(current_chunk)
                current_chunk = []

        if len(current_chunk) > 0:  # handles trailing tokens after the last delimiter
            chunks.append(current_chunk)

        total = sum(len(c) for c in chunks)
        assert total == len(token_ids), f"Sum of chunk lengths ({total}) ≠ original length ({len(token_ids)})"
        return chunks

    def merge_chunks(chunks, max_vineppo_chunks):
        if len(chunks) <= max_vineppo_chunks:
            return chunks
        else:
            # Will have to merge chunks
            max_chunks = max_vineppo_chunks - 1  # last one is left for pending tokens
            total_length_of_chunks = sum(len(chunk) for chunk in chunks)
            length_per_chunk = total_length_of_chunks // max_chunks
            chunk_ctr = 0
            merged_chunks = []
            while chunk_ctr < len(chunks):
                end_chunk_ctr = chunk_ctr
                current_chunk_length = 0
                while True:
                    current_chunk_length += len(chunks[end_chunk_ctr])
                    if current_chunk_length > length_per_chunk:
                        end_chunk_ctr += 1
                        break
                    end_chunk_ctr += 1
                    if end_chunk_ctr >= len(chunks):
                        break

                # Concatenate the chunks from chunk_ctr to end_chunk_ctr
                merged_chunks.append(list(itertools.chain(*chunks[chunk_ctr:end_chunk_ctr])))
                chunk_ctr = end_chunk_ctr

            last_chunk = list(itertools.chain(*chunks[chunk_ctr:]))
            if len(last_chunk) > 0:
                merged_chunks.append(last_chunk)
            total_length_of_merged_chunks = sum(len(chunk) for chunk in merged_chunks)
            assert (
                total_length_of_merged_chunks == total_length_of_chunks
            ), f"Total length of merged chunks {total_length_of_merged_chunks} does not match total length of chunks {total_length_of_chunks}"
            return merged_chunks

    def split_response(response_token_ids: List[int], split_indices: Optional[List[int]] = None, max_chunks: Optional[int] = None) -> List[int]:
        if split_indices is None:
            last_index = len(response_token_ids)
            step_boundaries = [0]
            if max_chunks is not None:
                chunk_size = math.ceil(last_index / max_chunks)
            else:
                chunk_size = 100
            cursor = 0
            while cursor < last_index:
                cursor += chunk_size
                if cursor >= last_index:
                    break
                step_boundaries.append(cursor)
        else:
            step_boundaries = sorted(list(set([0] + split_indices)))
            assert step_boundaries[0] == 0
        return step_boundaries

    def estimate_values_by_mc_rollouts(
        episodes_raw: List[Dict[str, Any]],
        split_indices: Optional[List[int]] = None,
        max_chunks: Optional[int] = None,
        metrics_dict: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        def get_response_prefixes(response_token_ids: List[int], states_for_value_estimation: List[int]) -> List[List[int]]:
            prefixes = []
            for state in states_for_value_estimation:
                prefixes.append(response_token_ids[:state])
            return prefixes

        def get_mc_queries(query_token_ids: List[int], response_prefixes_token_ids: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
            prefix_queries_token_ids = []
            max_tokens = []
            for token_ids in response_prefixes_token_ids:
                prefix_queries_token_ids.append(query_token_ids + token_ids)
                max_tokens.append(max(1, MAX_RESPONSE_TOKENS - len(token_ids)))  # this is how many response tokens are left.
            return prefix_queries_token_ids, max_tokens

        def get_value_estimates(sample: Dict[str, Any], response_prefixes: List[List[int]], mcs_token_ids: List[List[int]]) -> List[float]:
            values_estimates = []
            for prefix, mcs in zip(response_prefixes, mcs_token_ids):
                values = []
                for mc in mcs:
                    full_text = tokenizer.decode(prefix + mc, skip_special_tokens=False)
                    score, _ = compute_reward_fn(full_text, sample, EOS_TOKEN)
                    values.append(score)
                values_estimates.append(sum(values) / len(values))
            # Value estimate under the best-of-n policy
            values_estimates = [(1 - (1 - value_estimate) ** prover_policy_best_of_n) for value_estimate in values_estimates]

            return values_estimates

        def update_state_values(
            old_states: List[int],
            old_value_estimates: List[float],
            new_states: List[int],
            new_value_estimates: List[float],
        ) -> Tuple[List[int], List[float]]:
            values = {}
            for state, value_estimate in zip(old_states, old_value_estimates):
                values[state] = value_estimate
            for state, value_estimate in zip(new_states, new_value_estimates):
                assert state not in values
                values[state] = value_estimate
            sorted_states = sorted(values.keys())
            sorted_values = [values[state] for state in sorted_states]
            return sorted_states, sorted_values

        def extract_token_ids(vllm_outputs: List[CompletionOutput]) -> List[List[int]]:
            return [list(out.token_ids) for out in vllm_outputs]

        for idx, eps in enumerate(episodes_raw):
            eps["value_estimates"] = [eps["reward"]]
            eps["states"] = [len(eps["response_token_ids"])]
            eps["new_states"] = split_response(
                eps["response_token_ids"], split_indices[idx] if split_indices is not None else None, max_chunks=max_vineppo_chunks
            )

            # Get prefixes of from chunks of the response
            # i.e. for response [A, B, C, D, E] with new_states [0, 2, 4]
            # we get prefixes [[]], [A, B], [A, B, C, D]
            eps["mc_response_prefixes_token_ids"] = get_response_prefixes(eps["response_token_ids"], eps["new_states"])

            # Get MC queries where we just add the query to each prefix
            eps["mc_queries_token_ids"], eps["mc_queries_max_tokens"] = get_mc_queries(eps["query_token_ids"], eps["mc_response_prefixes_token_ids"])
            # Print a few samples
            if random.random() < 0.02:
                logger.info("********************Debugging TRAJECTORIES******************************")
                logger.info(f"Query: {tokenizer.decode(eps['query_token_ids'], skip_special_tokens=False)}")
                logger.info(f"Response: {tokenizer.decode(eps['response_token_ids'], skip_special_tokens=False)}")
                logger.info(f"New states: {eps['new_states']}")
                for idx, traj in enumerate(eps["mc_queries_token_ids"]):
                    logger.info(f"MC query {idx}: {tokenizer.decode(traj, skip_special_tokens=False)}")
                    logger.info(f"MC query max tokens {idx}: {eps['mc_queries_max_tokens'][idx]}")
                logger.info("**************************************************\n\n")

        # Flatten the MC queries to a single list which will be used for inference
        flatten_mc_queries = []
        flatten_mc_queries_max_tokens = []
        queries_count = []
        for eps in episodes_raw:
            flatten_mc_queries.extend(eps["mc_queries_token_ids"])
            flatten_mc_queries_max_tokens.extend(eps["mc_queries_max_tokens"])
            queries_count.append(len(eps["mc_queries_token_ids"]))

        # Auxiliary rollouts to get the value estimates
        logger.info("Monte-Carlo value estimation...")
        mc_outputs: List[RequestOutput] = inference_engine.generate(
            prompt_token_ids=flatten_mc_queries,
            sampling_params=[
                SamplingParams(
                    n=VINEPPO_K,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    max_tokens=max_tokens,
                    detokenize=False,
                    stop_token_ids=[EOS_TOKEN_ID],
                )
                for max_tokens in flatten_mc_queries_max_tokens
            ],
        )

        # Unflatten the MC rollouts
        # [
        #   // Episode 1
        #   [
        #     [tok1, tok2, ...],
        #     [tok1, tok2, ...],
        #     ...
        #   ],
        #   ...
        # ]
        unflattened_mc_token_ids: List[List[List[int]]] = []
        all_finish_reasons = [g.finish_reason for out in mc_outputs for g in out.outputs]
        proportion_finished = sum([fr == "stop" for fr in all_finish_reasons]) / len(all_finish_reasons)
        start = 0
        for count in queries_count:
            output_slice = mc_outputs[start : start + count]
            unflattened_mc_token_ids.append([extract_token_ids(out.outputs) for out in output_slice])
            start += count
        metrics_dict["value_estimation/proportion_finished"].append(proportion_finished)

        assert len(unflattened_mc_token_ids) == len(episodes_raw)

        # Compute the value estimates based on avg. MC returns
        for i, eps in enumerate(episodes_raw):
            mc_token_ids = unflattened_mc_token_ids[i]
            mc_value_estimates = get_value_estimates(
                sample=eps["sample"],
                response_prefixes=eps["mc_response_prefixes_token_ids"],
                mcs_token_ids=mc_token_ids,
            )
            eps["states"], eps["value_estimates"] = update_state_values(
                old_states=eps["states"],
                old_value_estimates=eps["value_estimates"],
                new_states=eps["new_states"],
                new_value_estimates=mc_value_estimates,
            )

        # Remove unnecessary keys
        for i, eps in enumerate(episodes_raw):
            eps.pop("new_states")
            eps.pop("mc_response_prefixes_token_ids")
            eps.pop("mc_queries_token_ids")
            eps.pop("mc_queries_max_tokens")

        return episodes_raw

    def get_tokens_advantages(states: List[int], value_estimates: List[float]) -> List[float]:
        tokens_advantages = []
        chunk_advantages = []
        chunk_lengths = []
        assert sorted(states) == states
        for i in range(len(states) - 1):
            length = states[i + 1] - states[i]
            advantage = value_estimates[i + 1] - value_estimates[i]
            tokens_advantages.extend([advantage] * length)
            chunk_advantages.append(advantage)
            chunk_lengths.append(length)
        return tokens_advantages, chunk_advantages, chunk_lengths

    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    # Process responses and calculate rewards
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE)) for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_query_token_ids, all_responses_token_ids, all_samples, all_rewards, orm_advantages = [], [], [], [], []

    for sample, group_indices in zip(samples, groups):
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        response_token_ids = [all_generations[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)

        rewards_and_metrics = [compute_reward_fn(resp, sample, EOS_TOKEN) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        all_rewards.extend(rewards)
        all_samples.extend([sample] * GENERATIONS_PER_SAMPLE)
        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)

        rewards_np = np.array(rewards)
        if LOSS_TYPE == "grpo":
            orm_advantage = (rewards_np - rewards_np.mean()) / (rewards_np.std() + 1e-4)
        elif LOSS_TYPE == "dr_grpo":
            orm_advantage = rewards_np - rewards_np.mean()
        else:
            raise ValueError(f"Unknown loss type: {LOSS_TYPE}")

        per_token_orm_advantage = [[adv] * len(resp) for adv, resp in zip(orm_advantage, response_token_ids)]
        orm_advantages.extend(per_token_orm_advantage)

        response_lengths = np.array([len(ids) for ids in response_token_ids])
        terminated_completions_mask = np.array([fr == "stop" for fr in finish_reasons], dtype=np.bool)
        non_terminated_completions_mask = ~terminated_completions_mask

        metrics_dict["extra/per_episode_rewards_list"].extend(rewards)  # used for logging in episode table
        metrics_dict["extra/per_episode_response_lengths_list"].extend(response_lengths.tolist())  # used for logging in episode table

        metrics_dict["rewards"].append(np.mean(rewards))
        metrics_dict["non_stop_rate"].append(np.mean([fr != "stop" for fr in finish_reasons]))
        metrics_dict["response_lengths"].append(np.mean([len(ids) for ids in response_token_ids]))

        metrics_dict["rewards_std"].append(np.array(rewards).std())
        metrics_dict["rewards_is_std_zero"].append(np.isclose(np.array(rewards).std(), 0.0))

        metrics_dict["response_lengths/mean_length"].append(response_lengths.mean())
        metrics_dict["response_lengths/min_length"].append(response_lengths.min())
        metrics_dict["response_lengths/max_length"].append(response_lengths.max())

        # log for dr grpo correctness checks
        correct_mask = np.array([rm["answer_reward"] > 0 for rm in reward_metrics], dtype=np.bool)
        incorrect_mask = ~correct_mask
        metrics_dict["extra/mean_correct_response_length"].append(response_lengths[correct_mask].mean() if correct_mask.any() else 0.0)
        metrics_dict["extra/mean_incorrect_response_length"].append(response_lengths[incorrect_mask].mean() if incorrect_mask.any() else 0.0)

        # just logging for sanity checks
        metrics_dict["extra/mean_terminated_length"].append(
            response_lengths[terminated_completions_mask].mean() if terminated_completions_mask.any() else 0.0
        )
        metrics_dict["extra/min_terminated_length"].append(
            response_lengths[terminated_completions_mask].min() if terminated_completions_mask.any() else 0.0
        )
        metrics_dict["extra/max_terminated_length"].append(
            response_lengths[terminated_completions_mask].max() if terminated_completions_mask.any() else 0.0
        )

        metrics_dict["extra/mean_non_terminated_length"].append(
            response_lengths[non_terminated_completions_mask].mean() if non_terminated_completions_mask.any() else 0.0
        )
        metrics_dict["extra/min_non_terminated_length"].append(
            response_lengths[non_terminated_completions_mask].min() if non_terminated_completions_mask.any() else 0.0
        )
        metrics_dict["extra/max_non_terminated_length"].append(
            response_lengths[non_terminated_completions_mask].max() if non_terminated_completions_mask.any() else 0.0
        )

        for rm in reward_metrics:
            for k, v in rm.items():
                metrics_dict.setdefault(f"reward_metrics/{k}", []).append(v)

    model_inputs = prepare_model_inputs(
        query_token_ids=all_query_token_ids,
        response_token_ids=all_responses_token_ids,
        device=device,
    )

    splits = None
    if top_k_entropy_tokens > 0:
        splits = []
        with torch.no_grad():
            # Note: This will be computed again in compute_pg_loss so there's a bit of redundancy here.
            # Need to do a forward pass over the model as VLLM doesn't provide support for returning all logits,
            # it only supports returning top-20 logits which is not enough for entropy calculation.
            # Also later when it's calculated it's calculated with gradient
            CHUNK_SIZE = 4
            for start_idx in range(0, len(model_inputs["input_ids"]), CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, len(model_inputs["input_ids"]))

                token_logits = compute_token_logits(policy_model, {k: v[start_idx:end_idx] for k, v in model_inputs.items()}, TEMPERATURE)
                token_entropy = compute_token_entropy(
                    token_logits, model_inputs["labels_mask"][start_idx:end_idx], make_query_tokens_entropy_negative=True
                )  # Shape: [batch_size, seq_len-1]

                indices = torch.topk(token_entropy, top_k_entropy_tokens, dim=-1).indices
                query_lens = torch.tensor([len(all_query_token_ids[i]) for i in range(start_idx, end_idx)], device=device).reshape(-1, 1)
                # Indexing w.r.t response tokens
                response_indexed_indices = indices - query_lens

                response_indexed_indices = response_indexed_indices.clamp(min=0)
                splits.extend(response_indexed_indices.tolist())
                del token_logits, query_lens, token_entropy

    elif split_and_merge_chunks > 0:
        splits = []
        num_chunks_before_merging = []
        num_chunks_after_merging = []
        for response_token_ids in all_responses_token_ids:
            delimiter_strings = [":\n\n", "\n\n"]
            delimiter_token_ids = [tokenizer.encode(s)[0] for s in delimiter_strings]
            chunks = split_tokens_at_delimiter(response_token_ids, delimiter_token_ids)
            num_chunks_before_merging.append(len(chunks))
            merged_chunks = merge_chunks(chunks, max_vineppo_chunks)
            assert (
                len(merged_chunks) <= max_vineppo_chunks
            ), f"Merged chunks should be less than or equal to max_vineppo_chunks: {len(merged_chunks)} <= {max_vineppo_chunks}"
            num_chunks_after_merging.append(len(merged_chunks))
            response_split_positions = []
            len_till_now = 0
            for idx, chunk in enumerate(merged_chunks):
                response_split_positions.append(len_till_now)
                len_till_now += len(chunk)
            splits.append(response_split_positions)
        metrics_dict["num_chunks_before_merging"].append(np.mean(num_chunks_before_merging))
        metrics_dict["num_chunks_after_merging"].append(np.mean(num_chunks_after_merging))

    raw_episodes: List[Dict[str, Any]] = []
    for i in range(len(all_samples)):
        eps = {
            "query_token_ids": all_query_token_ids[i],
            "response_token_ids": all_responses_token_ids[i],
            "sample": all_samples[i],
            "reward": all_rewards[i],
            "states": [len(all_responses_token_ids[i])],
            "value_estimates": [all_rewards[i]],
            "new_states": split_response(all_responses_token_ids[i], splits[i] if splits is not None else None),
        }
        raw_episodes.append(eps)

    raw_episodes = estimate_values_by_mc_rollouts(raw_episodes, splits, max_chunks=max_vineppo_chunks, metrics_dict=metrics_dict)

    all_advantages = []
    if top_k_entropy_tokens > 0:
        chunk_advantages = [[] for _ in range(top_k_entropy_tokens + 1)]
        chunk_lengths = [[] for _ in range(top_k_entropy_tokens + 1)]
    else:
        chunk_advantages = [[] for _ in range(max_vineppo_chunks)]
        chunk_lengths = [[] for _ in range(max_vineppo_chunks)]
    for idx, eps in enumerate(raw_episodes):
        progress_advantage, _chunk_advantages, _chunk_lengths = get_tokens_advantages(eps["states"], eps["value_estimates"])
        _orm_advantage = np.array(orm_advantages[idx])
        _progress_advantage = np.array(progress_advantage)
        final_advantage = (1 - prover_alpha) * _orm_advantage + prover_alpha * _progress_advantage

        # Add Advantage alignment monitoring
        sign_orm_advantage = np.sign(_orm_advantage)
        sign_progress_advantage = np.sign(_progress_advantage)
        sign_final_advantage = np.sign(final_advantage)

        # Average across all tokens
        metrics_dict["extra/frac_positive_orm_advantage"].append(np.mean(sign_orm_advantage == 1))
        metrics_dict["extra/frac_positive_progress_advantage"].append(np.mean(sign_progress_advantage == 1))
        metrics_dict["extra/frac_positive_final_advantage"].append(np.mean(sign_final_advantage == 1))
        metrics_dict["extra/frac_negative_orm_advantage"].append(np.mean(sign_orm_advantage == -1))
        metrics_dict["extra/frac_negative_progress_advantage"].append(np.mean(sign_progress_advantage == -1))
        metrics_dict["extra/frac_negative_final_advantage"].append(np.mean(sign_final_advantage == -1))
        metrics_dict["extra/frac_zero_orm_advantage"].append(np.mean(sign_orm_advantage == 0))
        metrics_dict["extra/frac_zero_progress_advantage"].append(np.mean(sign_progress_advantage == 0))
        metrics_dict["extra/frac_zero_final_advantage"].append(np.mean(sign_final_advantage == 0))

        # Alignment levels averaged across all tokens
        advantage_alignment = _orm_advantage * _progress_advantage
        inner_product_advantage_alignment = np.dot(_orm_advantage, _progress_advantage)

        sign_advantage_alignment = np.sign(advantage_alignment)
        metrics_dict["extra/frac_positive_advantage_alignment"].append(np.mean(sign_advantage_alignment == 1))
        metrics_dict["extra/frac_negative_advantage_alignment"].append(np.mean(sign_advantage_alignment == -1))
        metrics_dict["extra/frac_zero_advantage_alignment"].append(np.mean(sign_advantage_alignment == 0))
        metrics_dict["extra/orm_adv_progress_adv_inner_product"].append(inner_product_advantage_alignment)
        metrics_dict["extra/total_length"].append(len(final_advantage))

        all_advantages.append(list(final_advantage))

        for chunk_idx, _chunk_advantage in enumerate(_chunk_advantages):
            chunk_advantages[chunk_idx].append(_chunk_advantage)
            chunk_lengths[chunk_idx].append(_chunk_lengths[chunk_idx])

    # We will take mean of only chunks that exist, note that chunk_advantages is a list where each element may have different number of elements
    # Number of max_chunks * batch_size (could be lesser than that if some chunks are not present)
    for chunk_indx in range(max(max_vineppo_chunks, top_k_entropy_tokens + 1)):
        metrics_dict[f"extra/mean_chunk_{chunk_indx}_advantage"].append(np.array(chunk_advantages[chunk_indx]).mean())
        metrics_dict[f"extra/mean_chunk_{chunk_indx}_length"].append(np.array(chunk_lengths[chunk_indx]).mean())

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    model_inputs = update_model_inputs_with_advantages(
        query_token_ids=all_query_token_ids,
        response_token_ids=all_responses_token_ids,
        advantages=all_advantages,
        model_inputs=model_inputs,
        device=device,
    )

    return episodes, model_inputs


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: torch.Tensor,
    LOSS_TYPE: str,
    MAX_RESPONSE_TOKENS: int,
    TEMPERATURE: float,
    KL_COEFFICIENT: float,
    CLIP_EPSILON: float = 0.1,
    metrics_dict: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    input_ids = batch["input_ids"]  # [B, L]
    attention_mask = batch["attention_mask"]  # [B, L]
    labels = batch["labels"]  # [B, L]
    completion_mask = batch["labels_mask"]  # [B, L]
    advantages = batch["advantages"]  # [B, L]
    ref_per_token_logps = batch["ref_log_probs"]

    if "old_log_probs" in batch:
        old_per_token_logps = batch["old_log_probs"]
    else:
        old_per_token_logps = None

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels_mask": completion_mask,
    }

    per_token_logps = compute_token_log_probs(policy_model, model_inputs, TEMPERATURE)
    completion_mask = completion_mask[..., 1:].to(per_token_logps.dtype)

    if old_per_token_logps is None:
        # performing on-policy updates
        policy_loss = -per_token_logps * advantages[..., 1:]  # [B, L-1]
        policy_loss = policy_loss * completion_mask
    else:
        # performing off-policy updates (or 𝜇 iterations) -- look at algorithm 1 in https://arxiv.org/abs/2402.03300
        # need to calculate importance sampling ratio using old_logps
        # inspired by: https://github.com/volcengine/verl/blob/7dc3ee7476c9af1042636f44bca9e412d1de1a6d/verl/trainer/ppo/core_algos.py#L533

        # ratio
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # importance sampling ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)

        # loss (minimum of clipped and unclipped objective)
        unclipped = ratio * advantages[..., 1:]  # [B, L-1]
        clipped = clipped_ratio * advantages[..., 1:]  # [B, L-1]

        policy_loss = -torch.min(unclipped, clipped)
        policy_loss = policy_loss * completion_mask  # mask padded tokens

        metrics_dict["clip_ratio"].append(((ratio * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)).item())

    # KL penalty
    # J. Schulman. Approximating kl divergence, 2020.
    # http://joschu.net/blog/kl-approx.html

    ref_logratio = ref_per_token_logps - per_token_logps
    kl_penalty = torch.exp(ref_logratio) - 1 - ref_logratio  # [B, L-1]
    kl_penalty = kl_penalty * completion_mask  # [B, L-1]

    loss = policy_loss + KL_COEFFICIENT * kl_penalty  # [B, L-1]

    # collate loss
    if LOSS_TYPE == "grpo":
        # each response's loss gets divided by its length, then averaged across the batch
        # this is not correct, as mentioned in Dr. GRPO paper
        loss = (loss.sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
    elif LOSS_TYPE == "dr_grpo":
        loss = loss.sum() / (loss.shape[0] * MAX_RESPONSE_TOKENS)  # should be same as total_response_len
    else:
        raise ValueError(f"Unknown loss type: {LOSS_TYPE}")
    with torch.no_grad():
        entropy = -per_token_logps.sum().item() / (per_token_logps.shape[0] * MAX_RESPONSE_TOKENS)
        zero_advantages_ratio = close_to_zero(advantages[..., 1:], completion_mask).item() / (completion_mask.sum().item())

    metrics_dict["policy_loss"].append(loss.item())
    metrics_dict["kl_penalty"].append(kl_penalty.sum().item() / (kl_penalty.shape[0] * MAX_RESPONSE_TOKENS))
    metrics_dict["entropy"].append(entropy)  # divide by constant
    metrics_dict["zero_advantages_ratio"].append(zero_advantages_ratio)  # scalar

    return loss


def load_dataset(task: str, tokenizer: AutoTokenizer, model_name: str):
    dataset = get_dataset(task)
    if dist.get_rank() != 0:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    logger.info(f"The task is {task}")
    train_dataset = dataset["train"].map(
        preprocess_example,
        num_proc=6,
        fn_kwargs={
            "task": task,
            "tokenizer": tokenizer,
            "model_name": model_name,
        },
        desc="Preprocessing Train dataset",
        load_from_cache_file=False,
    )
    test_datasets = {}
    for test_dataset_name, test_dataset in dataset["test"].items():
        test_datasets[test_dataset_name] = test_dataset.map(
            preprocess_example,
            num_proc=6,
            fn_kwargs={
                "task": task,
                "tokenizer": tokenizer,
                "model_name": model_name,
            },
            desc=f"Preprocessing Test dataset {test_dataset_name}",
            load_from_cache_file=False,  # done to ensure that any changes to the prompt template are reflected in the dataset
        )

    if dist.get_rank() == 0:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.barrier(device_ids=[torch.cuda.current_device()])

    return train_dataset, test_datasets


def main(rank: int):
    config = tyro.cli(GRPOConfig)
    logger.info("The config is: ", config)

    nproc = int(os.environ.get("WORLD_SIZE", "1"))
    nproc = config.num_processes
    initialize_training_process_group(rank, nproc)
    curr_cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    print("curr_cuda_device: ", curr_cuda_device)
    if dist.get_rank() != 0:
        logger.setLevel(logging.ERROR)

    ############################################
    # Hyperparameters
    ############################################

    # Model configuration
    MODEL_NAME = config.model_name

    # RL parameters
    # Total number of training iterations
    NUM_ITERATIONS = config.num_iterations
    # Number of episodes to collect per iteration for training
    logger.info(f"Number of ranks: {dist.get_world_size()}")

    GENERATIONS_PER_SAMPLE = config.num_generations

    EPISODES_PER_ITERATION = (
        config.per_device_train_batch_size * config.num_generations * config.num_processes
    )  # Num samples * num generations * num ranks
    EPISODES_PER_ITERATION_PER_RANK = EPISODES_PER_ITERATION // config.num_processes
    # Number of responses to generate for each input prompt

    # Controls how much the policy can deviate from the reference model
    KL_COEFFICIENT = config.kl_coeff

    LOSS_TYPE = config.loss_type  # choose b/w ["grpo", "dr_grpo"]

    # off-policy PPO updates
    PPO_EPOCHS = config.num_ppo_epochs  # defaults to 1 -- Same as μ in https://arxiv.org/abs/2402.03300 in Algorithm 1.
    CLIP_EPSILON = config.epsilon

    # Training hyperparameters
    # Batch size for each GPU device during training
    PER_DEVICE_TRAIN_BATCH_SIZE = config.per_device_ppo_micro_batch_size
    assert EPISODES_PER_ITERATION_PER_RANK % PER_DEVICE_TRAIN_BATCH_SIZE == 0

    # Learning rate for model updates
    LEARNING_RATE = config.learning_rate

    # Sampling parameters
    # Maximum number of tokens to generate in each response
    MAX_RESPONSE_TOKENS = config.max_response_tokens
    # Controls randomness in generation (higher = more random)
    TEMPERATURE = config.temperature
    # Nucleus sampling parameter (1.0 = disabled)
    # to avoid sampling unused tokens absent from tokenizer see https://github.com/vllm-project/vllm/issues/13175#issuecomment-2781842571
    TOP_P = config.top_p
    # Top-k sampling parameter (-1 = disabled)
    TOP_K = config.top_k  # no top k
    # DeepSpeed configuration
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": False},
        "train_batch_size": config.per_device_ppo_mini_batch_size * config.num_generations * config.num_processes,  # effective ppo batch size
        "train_micro_batch_size_per_gpu": config.per_device_ppo_micro_batch_size,
        "gradient_accumulation_steps": (config.per_device_ppo_mini_batch_size * config.num_generations) // config.per_device_ppo_micro_batch_size,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True,
                "fused": True,
            },
        },
    }
    ref_deepspeed_config = {
        "bf16": {"enabled": True},
        # No effect
        "train_batch_size": config.per_device_ppo_mini_batch_size * config.num_generations * config.num_processes,  # effective ppo batch size
        "train_micro_batch_size_per_gpu": config.per_device_ppo_micro_batch_size,
        "gradient_accumulation_steps": (config.per_device_ppo_mini_batch_size * config.num_generations) // config.per_device_ppo_micro_batch_size,
    }
    compute_reward_fn = get_reward_fn(config.task)

    dist.barrier(device_ids=[torch.cuda.current_device()])

    model_name_short = MODEL_NAME.split("/")[-1]
    if config.run_id is None:
        RUN_NAME = (
            f"{model_name_short}_temp-{TEMPERATURE}_kl-{KL_COEFFICIENT}"
            f"_lr-{LEARNING_RATE}_al-{config.algorithm}"
            f"_task-{config.task}_loss-{config.loss_type}"
        )
    else:
        RUN_NAME = config.run_id
    EXP_DIR = Path(config.output_dir) / RUN_NAME
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    ############################################
    # Prompts and Dataset
    ############################################

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    # Prover tokenizer
    if config.prover_policy_model_name is not None:
        prover_tokenizer = AutoTokenizer.from_pretrained(config.prover_policy_model_name)
        EOS_TOKEN_ID = prover_tokenizer.eos_token_id
        EOS_TOKEN = prover_tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)
    else:
        prover_tokenizer = None

    train_dataset, test_datasets = load_dataset(config.task, tokenizer, MODEL_NAME)
    orig_train_dataset_size = len(train_dataset)

    # Shard the training dataset
    train_dataset = train_dataset.shard(num_shards=dist.get_world_size(), index=dist.get_rank())

    logger.info(f"Train dataset size: {orig_train_dataset_size}; each rank will process {len(train_dataset)} samples")
    for test_dataset_name, test_dataset in test_datasets.items():
        logger.info(f"Test dataset {test_dataset_name} size: {len(test_dataset)}")

    ############################################
    # Initialize Models
    ############################################

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    prover_model = None
    if config.prover_policy_model_name is not None:
        prover_model = AutoModelForCausalLM.from_pretrained(
            config.prover_policy_model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=torch.cuda.current_device(),
        )

    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    see_memory_usage("Before initializing DeepSpeed engines", force=dist.get_rank() == 0)

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )

    reference_model.module.cpu()

    if config.prover_policy_model_name is not None:
        # Keeping the prover model on GPU for now. Maybe I need to move to CPU if OOM.
        prover_model, *_ = deepspeed.initialize(
            model=prover_model,
            config=ref_deepspeed_config,  # since the reference model is not used for training I'm sticking to it's config.
            model_parameters=prover_model.parameters(),
        )
        prover_model.eval()

    dist.barrier(device_ids=[torch.cuda.current_device()])

    ############################################
    # Initialize vLLM (Inference) engine
    ############################################

    see_memory_usage("Before initializing inference engine", force=dist.get_rank() == 0)

    if dist.get_rank() != 0:
        # Disable root vllm logger for non-main ranks
        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.ERROR)

    # skip_tokenizer_init=False so LLM does not complain about the tokens present in the model but not in the tokenizer
    # (see https://github.com/vllm-project/vllm/issues/13175), remove when fixed in vllm or qwen.
    inference_engine = LLM(
        model=MODEL_NAME,
        skip_tokenizer_init=False,
        gpu_memory_utilization=0.15 if not config.current_policy_as_prover else 0.35,
        enable_prefix_caching=True,
        swap_space=4,
        scheduling_policy="fcfs",
        # preemption_mode="swap",
        dtype=torch.bfloat16,
        max_model_len=config.max_response_tokens + config.max_prompt_tokens,
        # enable_sleep_mode=True,
        device=f"cuda:{torch.cuda.current_device()}",
        tensor_parallel_size=1,
    )

    if config.algorithm == "vineppo":
        logits_processors = [fix_oov_logits_processor(inference_engine)]
    else:
        logits_processors = None

    prover_inference_engine = None
    prover_logits_processors = None

    see_memory_usage("After initializing inference engine", force=dist.get_rank() == 0)

    # Wandb for logging. Only rank 0 will initialize wandb
    if dist.get_rank() == 0:
        wandb.init(
            project="rl-reasoning",
            name=RUN_NAME,
            resume="allow",
            config=asdict(config),
        )

    sampler_rng = np.random.default_rng(seed=42)
    NUM_SAMPLES_PER_ITERATION = EPISODES_PER_ITERATION_PER_RANK // GENERATIONS_PER_SAMPLE

    # Load checkpoint if it exists
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None and config.load_checkpoint:
        logger.info(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        begin_iter = ckpt_iter + 1
        load_model_into_vllm(policy_model, inference_engine)

        logger.info(f"Skipping {ckpt_iter} rounds of samples")
        for _ in trange(ckpt_iter, disable=dist.get_rank() != 0):
            _ = sampler_rng.choice(
                len(train_dataset),
                size=NUM_SAMPLES_PER_ITERATION,
                replace=False,
            )

    for iteration in trange(begin_iter, NUM_ITERATIONS):
        logger.info(f"Iteration {iteration}/{NUM_ITERATIONS}")

        metrics_dict = defaultdict(list)

        #########################################################
        # Evaluation
        #########################################################

        eval_stats = None
        if (iteration % 20 == 0 and iteration > 0) and dist.get_rank() == 0:  # Only rank 0 will evaluate:
            eval_stats = {}
            for test_dataset_name, test_dataset in test_datasets.items():
                logger.info(f"Evaluating on {test_dataset_name}...")
                _eval_episodes, _eval_stats = evaluate_on_test_set(
                    inference_engine=inference_engine,
                    test_dataset=test_dataset,
                    tokenizer=tokenizer,
                    eos_token=EOS_TOKEN,
                    eval_sampling_params=SamplingParams(
                        temperature=config.temperature,
                        max_tokens=config.max_eval_tokens,
                        n=config.num_generations_eval,
                        detokenize=False,
                        stop_token_ids=[EOS_TOKEN_ID],
                    ),
                    reward_func=lambda completion, sample: compute_reward_fn(completion, sample, EOS_TOKEN),
                )
                _eval_episode_table = dump_episodes(
                    episodes=_eval_episodes,
                    episodes_stats=_eval_stats,
                    exp_dir=EXP_DIR,
                    tokenizer=tokenizer,
                    iteration=iteration,
                    is_eval=True,
                    eval_dataset_name=test_dataset_name,
                )
                for k, v in _eval_stats.items():
                    eval_stats[f"{test_dataset_name}/{k}"] = v
                wandb.log({f"eval/{test_dataset_name}/episodes": _eval_episode_table, "iteration": iteration})

        dist.barrier(device_ids=[torch.cuda.current_device()])

        #########################################################
        # Generate Episodes
        #########################################################

        # Sample training batch
        indices = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_ITERATION, replace=False)
        samples = train_dataset.select(indices)

        gen_time = time.time()

        # Sample responses from the policy model
        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=GENERATIONS_PER_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_RESPONSE_TOKENS,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
                logits_processors=logits_processors,
            ),
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]

        logger.info(f"Generated {len(all_generations)} responses")
        logger.info(f"Time taken to generate {len(all_generations)} responses: {time.time() - gen_time} seconds")

        # Process responses and calculate rewards
        if config.algorithm == "grpo":
            episodes, model_inputs = create_training_episodes_and_model_inputs(
                samples=samples,
                all_generations=all_generations,
                all_finish_reasons=all_finish_reasons,
                tokenizer=tokenizer,
                EOS_TOKEN=EOS_TOKEN,
                GENERATIONS_PER_SAMPLE=GENERATIONS_PER_SAMPLE,
                compute_reward_fn=compute_reward_fn,
                LOSS_TYPE=LOSS_TYPE,
                metrics_dict=metrics_dict,
                device=curr_cuda_device,
            )
        elif config.algorithm == "vineppo":
            if prover_model is not None:
                if prover_inference_engine is None:
                    if config.current_policy_as_prover:
                        logger.info("Using Current Policy as the Prover")
                        prover_inference_engine = inference_engine
                        prover_logits_processors = logits_processors
                    else:
                        time_to_initialize_prover_inference_engine = time.time()
                        prover_inference_engine = LLM(
                            model=config.prover_policy_model_name,
                            skip_tokenizer_init=False,
                            gpu_memory_utilization=0.25,
                            enable_prefix_caching=True,
                            swap_space=24,
                            scheduling_policy="fcfs",
                            # preemption_mode="swap",
                            dtype=torch.bfloat16,
                            max_model_len=config.max_response_tokens + config.max_prompt_tokens,
                            # enable_sleep_mode=False,
                            device=f"cuda:{torch.cuda.current_device()}",
                            tensor_parallel_size=1,
                        )
                        prover_logits_processors = fix_oov_logits_processor(prover_inference_engine)
                        logger.info(
                            f"Time taken to initialize prover inference engine: {time.time() - time_to_initialize_prover_inference_engine} seconds"
                        )

            else:
                prover_inference_engine = inference_engine
                prover_logits_processors = logits_processors

            logger.info(f"Prover inference engine logits processors: {prover_logits_processors}")
            episodes, model_inputs = create_vineppo_training_episodes_and_model_inputs(
                inference_engine=prover_inference_engine,
                samples=samples,
                all_generations=all_generations,
                all_finish_reasons=all_finish_reasons,
                tokenizer=tokenizer,
                EOS_TOKEN_ID=EOS_TOKEN_ID,
                EOS_TOKEN=EOS_TOKEN,
                GENERATIONS_PER_SAMPLE=GENERATIONS_PER_SAMPLE,
                MAX_RESPONSE_TOKENS=MAX_RESPONSE_TOKENS,
                VINEPPO_K=config.vineppo_k,
                TEMPERATURE=TEMPERATURE,
                TOP_P=TOP_P,
                TOP_K=TOP_K,
                compute_reward_fn=compute_reward_fn,
                metrics_dict=metrics_dict,
                device=curr_cuda_device,
                policy_model=policy_model,  # TODO: Maybe I need to use the prover policy here?
                top_k_entropy_tokens=config.top_k_entropy_tokens,
                max_vineppo_chunks=config.max_vineppo_chunks,
                prover_policy_best_of_n=config.prover_policy_best_of_n,
                prover_alpha=config.prover_alpha,
                LOSS_TYPE=LOSS_TYPE,
                split_and_merge_chunks=config.split_and_merge_chunks,
            )

        else:
            raise ValueError(f"Invalid algorithm: {config.algorithm}")

        # inference_engine.sleep(1)
        # gc.collect()
        # torch.cuda.empty_cache()
        # time.sleep(1)

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=metrics_dict,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer,
            iteration=iteration,
            do_save=iteration % 10 == 0 or iteration == 0,
        )

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        logger.info("Moving reference model to GPU")
        reference_model.module.to(curr_cuda_device)
        reference_model.eval()

        with torch.no_grad():
            ref_log_probs = []
            for i in trange(
                0,
                EPISODES_PER_ITERATION_PER_RANK,
                config.per_device_ppo_micro_batch_size,
                desc="Computing reference logprobs",
                disable=dist.get_rank() != 0,
            ):
                batch = {k: v[i : i + config.per_device_ppo_micro_batch_size] for k, v in model_inputs.items()}
                ref_log_probs.append(
                    compute_token_log_probs(
                        model=reference_model,
                        inputs=batch,
                        temperature=TEMPERATURE,
                    )
                )
            ref_log_probs = torch.cat(ref_log_probs)
            model_inputs["ref_log_probs"] = ref_log_probs
            del ref_log_probs

        # cache old log_probs for importance sampling for off-policy updates
        if (config.per_device_train_batch_size != config.per_device_ppo_mini_batch_size) or (
            config.num_ppo_epochs > 1
        ):  # if doing off-policy updates (or μ iterations), cache current model logprobs
            with torch.no_grad():
                old_log_probs = []
                for i in trange(
                    0,
                    EPISODES_PER_ITERATION_PER_RANK,
                    PER_DEVICE_TRAIN_BATCH_SIZE,
                    desc="Computing logprobs for importance sampling",
                    disable=dist.get_rank() != 0,
                ):
                    batch = {k: v[i : i + PER_DEVICE_TRAIN_BATCH_SIZE] for k, v in model_inputs.items()}
                    old_log_probs.append(
                        compute_token_log_probs(
                            model=policy_model,
                            inputs=batch,
                            temperature=TEMPERATURE,
                        )
                    )
                old_log_probs = torch.cat(old_log_probs)
                model_inputs["old_log_probs"] = old_log_probs
                del old_log_probs

        # Free memory taken by reference model
        logger.info("Moving reference model back to CPU")
        reference_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # Calculate losses and update model
        policy_model.train()
        total_response_len = (model_inputs["labels"] != -100).sum()
        train_time = time.time()

        effective_per_device_train_batch_size = config.per_device_train_batch_size * config.num_generations
        effective_per_device_ppo_mini_batch_size = config.per_device_ppo_mini_batch_size * config.num_generations

        for _ in trange(PPO_EPOCHS, desc="μ iteration", disable=dist.get_rank() != 0):
            for idx in trange(
                0,
                effective_per_device_train_batch_size,
                effective_per_device_ppo_mini_batch_size,
                desc="PPO mini batch",
                disable=dist.get_rank() != 0,
            ):
                ppo_mini_batch = {k: v[idx : idx + effective_per_device_ppo_mini_batch_size] for k, v in model_inputs.items()}

                # perform grad accum over ppo_mini_batch
                for jdx in trange(
                    0,
                    effective_per_device_ppo_mini_batch_size,
                    config.per_device_ppo_micro_batch_size,
                    desc="Gradient accum for PPO mini batch",
                    disable=dist.get_rank() != 0,
                ):
                    ppo_micro_batch = {k: v[jdx : jdx + config.per_device_ppo_micro_batch_size] for k, v in ppo_mini_batch.items()}
                    # Compute policy gradient loss
                    loss = compute_pg_loss(
                        policy_model=policy_model,
                        batch=ppo_micro_batch,
                        LOSS_TYPE=LOSS_TYPE,
                        MAX_RESPONSE_TOKENS=MAX_RESPONSE_TOKENS,
                        total_response_len=total_response_len,
                        TEMPERATURE=TEMPERATURE,
                        KL_COEFFICIENT=KL_COEFFICIENT,
                        CLIP_EPSILON=CLIP_EPSILON,
                        metrics_dict=metrics_dict,
                    )

                    # Track metrics
                    metrics_dict["loss"].append(loss.item())
                    grad_norm = policy_model.get_global_grad_norm()
                    if grad_norm is not None:
                        grad_norm = grad_norm.item()
                    metrics_dict["grad_norm"].append(grad_norm)

                    # Backpropagation and optimization step
                    # scale_wrt_gas=False because we are already normalizing by total_response_len
                    policy_model.backward(loss, scale_wrt_gas=False)

                    # this will take a gradient step at the last iteration of the loop
                    # before that it will just accumulate gradients
                    policy_model.step()

        logger.info(f"Time taken to train: {time.time() - train_time} seconds")

        #########################################################
        # Update inference engine weights
        #########################################################

        # gc.collect()
        # torch.cuda.empty_cache()
        # time.sleep(1)

        # inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        #########################################################
        # Log metrics
        #########################################################

        if dist.get_rank() == 0:
            train_metrics = {k: np.mean(v) for k, v in metrics_dict.items() if None not in v}
            train_metrics["learning_rate"] = policy_model.get_lr()[0]
            logs = {
                "iteration": iteration,
                f"episodes/iter_{iteration:06d}": episode_table,
                **{f"{k}": v for k, v in train_metrics.items()},
            }
            if eval_stats is not None:
                logs.update({f"eval/{k}": np.mean(v) for k, v in eval_stats.items()})
            wandb.log(logs)

            selected_keys = [
                "train/kl_penalty",
                "train/rewards",
                "train/reward_metrics/format_reward",
                "train/reward_metrics/answer_reward",
                "train/response_lengths",
                "eval/rewards",
                "eval/reward_metrics/format_reward",
                "eval/reward_metrics/answer_reward",
            ]
            selected_metrics = {k: float(logs[k]) for k in selected_keys if k in logs}
            logger.info(f"KEY METRICS: {selected_metrics}")

        if iteration % 10 == 0 and iteration != 0 and config.save_checkpoint:
            logger.info("Saving hf model")
            ckpt_dir = EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}"

            logger.info("Saving HF model")
            if dist.get_rank() == 0:
                policy_model.module.save_pretrained(str(ckpt_dir / "hf_model"))
                tokenizer.save_pretrained(str(ckpt_dir / "hf_model"))
            dist.barrier(device_ids=[torch.cuda.current_device()])

            logger.info("Saving DeepSpeed checkpoint")
            policy_model.save_checkpoint(str(ckpt_dir / "deepspeed"))

            if dist.get_rank() == 0:
                clean_up_checkpoints(
                    exp_dir=EXP_DIR,
                    keep_every_n_steps=None,  # remove all but the latest checkpoint
                    exclude=[ckpt_dir],
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])

    dist.destroy_process_group()


if __name__ == "__main__":
    config = tyro.cli(GRPOConfig)

    n_gpus = torch.cuda.device_count()
    if config.num_processes > n_gpus:
        raise ValueError(f"Requested {config.num_processes} processes, but only {n_gpus} GPUs are available.")

    if config.num_processes == 1:
        main(rank=0)
    else:
        torch.multiprocessing.spawn(main, nprocs=config.num_processes)