"""
Implementation for Best-of-N aware finetuning (https://openreview.net/forum?id=77gQUdQhE7 , https://arxiv.org/abs/2412.15287)
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

# ─── Standard library ──────────────────────────────────────────────────────────
import gc
import logging
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
from vllm import LLM, SamplingParams

import wandb

# ─── Local application imports ─────────────────────────────────────────────────
from tasks import get_dataset, get_reward_fn, preprocess_example
from utils import (
    clean_up_checkpoints,
    close_to_zero,
    compute_token_log_probs,
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
    # kl_coeff: float = 0.001

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    task: str = ""
    output_dir: Optional[str] = "results"  # required field

    run_id: Optional[str] = "Qwen2.5-1.5B-Instruct-Train"
    load_checkpoint: Optional[bool] = True
    save_checkpoint: Optional[bool] = True

    debug: Optional[bool] = False

    loss_type: str = "dr_grpo"  # choose b/w ["grpo", "dr_grpo"]
    algorithm: str = "grpo"  # we use this just to name the experiment folder

    epsilon: Optional[float] = 0.2  # Epsilon value for clipping

    # sampling parameters
    temperature: float = 0.6
    top_p: float = 0.999  # to avoid sampling unused tokens absent from
    top_k: int = -1  # no top k
    seed: int = 65

    # inference aware clipping for -ve samples
    best_of_n_clip_positive: float = 3.0
    best_of_n_clip_negative: float = 3.0
    kl_schedule: str = "constant"  # choose from ["constant", "linear"]
    kl_coeff: float = 0.001
    initial_kl_coeff: float = 0.001
    final_kl_coeff: float = 0.001

    def __post_init__(self):

        assert self.output_dir is not None

        if self.per_device_ppo_mini_batch_size is None:
            self.per_device_ppo_mini_batch_size = self.per_device_train_batch_size
        else:
            assert self.per_device_train_batch_size % self.per_device_ppo_mini_batch_size == 0

        assert (self.per_device_ppo_mini_batch_size * self.num_generations) % self.per_device_ppo_micro_batch_size == 0


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
    best_of_n_clip_positive: float = None,
    best_of_n_clip_negative: float = None,
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
        elif LOSS_TYPE == "best_of_n":
            # https://openreview.net/forum?id=77gQUdQhE7
            # https://arxiv.org/abs/2412.15287
            # we implement Lemma 3 that works with Binary rewards
            # equation 9

            p_fail = 1 - rewards.mean()
            N = GENERATIONS_PER_SAMPLE

            # Lemma 3 in https://arxiv.org/abs/2412.15287
            correct_weight = (N * (p_fail ** (N - 1))) / (1 - (p_fail**N) + 1e-8)
            incorrect_weight = -(N * p_fail) / (1 - p_fail + 1e-8)

            # should probably clip this -- as similarly done in TOPR paper https://arxiv.org/abs/2503.14286
            correct_weight = np.clip(correct_weight, 0, best_of_n_clip_positive)
            incorrect_weight = np.clip(incorrect_weight, -best_of_n_clip_negative, 0)

            correct_mask = rewards > 0
            assert (rewards[correct_mask] == 1.0).all()  # we assume we don't have any format rewards for now

            advantages = np.zeros_like(rewards)
            advantages[correct_mask] = correct_weight
            advantages[~correct_mask] = incorrect_weight

            # also log weights
            metrics_dict["bon/best_of_n_p_fail"].append(p_fail)
            metrics_dict["bon/best_of_n_correct_weight"].append(correct_weight)
            metrics_dict["bon/best_of_n_incorrect_weight"].append(incorrect_weight)
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
    elif LOSS_TYPE == "dr_grpo" or LOSS_TYPE == "best_of_n":
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
    # Rank 0 will preprocess the dataset first
    if dist.get_rank() != 0:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    logger.info("The task is", task)
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
    train_dataset = train_dataset.shuffle(seed=42)
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
        ).shuffle(seed=42)

    if dist.get_rank() == 0:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.barrier(device_ids=[torch.cuda.current_device()])

    return train_dataset, test_datasets


def main(rank: int):
    config = tyro.cli(GRPOConfig)
    logger.info(config)

    nproc = int(os.environ.get("WORLD_SIZE", "1"))
    nproc = config.num_processes
    initialize_training_process_group(rank, nproc)
    curr_cuda_device = torch.device("cuda")

    # Disable logging for non-main processes to avoid duplicate logs
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
    # KL_COEFFICIENT = config.kl_coeff # set later per iteration

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
            f"{model_name_short}_temp-{TEMPERATURE}" f"_lr-{LEARNING_RATE}_al-{config.algorithm}" f"_task-{config.task}_loss-{config.loss_type}"
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
        gpu_memory_utilization=0.35,
        enable_prefix_caching=True,
        swap_space=4,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=config.max_response_tokens + config.max_prompt_tokens,
        enable_sleep_mode=True,
        device=f"cuda:{torch.cuda.current_device()}",
        tensor_parallel_size=1,
    )

    if config.algorithm == "vineppo":
        logits_processors = [fix_oov_logits_processor(inference_engine)]
    else:
        logits_processors = None

    see_memory_usage("After initializing inference engine", force=dist.get_rank() == 0)

    # Wandb for logging. Only rank 0 will initialize wandb
    if dist.get_rank() == 0:
        wandb.init(
            project="rl-reasoning",
            name=RUN_NAME,
            resume="allow",
            config=asdict(config),
        )

    sampler_rng = np.random.default_rng(seed=config.seed)
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

    if config.kl_schedule == "linear":
        kl_coeffs = np.linspace(config.initial_kl_coeff, config.final_kl_coeff, NUM_ITERATIONS).tolist()
    else:
        kl_coeffs = [config.kl_coeff] * NUM_ITERATIONS

    for iteration in trange(begin_iter, NUM_ITERATIONS):
        logger.info(f"Iteration {iteration}/{NUM_ITERATIONS}")

        metrics_dict = defaultdict(list)

        KL_COEFFICIENT = kl_coeffs[iteration]
        metrics_dict["kl_coefficient"].append(KL_COEFFICIENT)
        print(f"Using KL coefficient for iteration {iteration}:", KL_COEFFICIENT)

        #########################################################
        # Evaluation
        #########################################################

        eval_stats = None
        if (iteration % 50 == 0 and iteration > 0) and dist.get_rank() == 0:  # Only rank 0 will evaluate:
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

        # Sample responses
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
                best_of_n_clip_positive=config.best_of_n_clip_positive,
                best_of_n_clip_negative=config.best_of_n_clip_negative,
            )
        else:
            raise ValueError(f"Invalid algorithm: {config.algorithm}")

        inference_engine.sleep(1)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

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

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
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
                "kl_penalty",
                "kl_coefficient",
                "rewards",
                "reward_metrics/format_reward",
                "reward_metrics/answer_reward",
                "response_lengths",
                "eval/rewards",
                "eval/reward_metrics/format_reward",
                "eval/reward_metrics/answer_reward",
            ]
            selected_metrics = {k: float(logs[k]) for k in selected_keys if k in logs}
            logger.info(f"KEY METRICS: {selected_metrics}")

        if iteration % 25 == 0 and iteration != 0 and config.save_checkpoint:
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
    # args = arg_parser.parse_args()
    config = tyro.cli(GRPOConfig)

    n_gpus = torch.cuda.device_count()
    if config.num_processes > n_gpus:
        raise ValueError(f"Requested {config.num_processes} processes, but only {n_gpus} GPUs are available.")

    if config.num_processes == 1:
        main(rank=0)
    else:
        torch.multiprocessing.spawn(main, nprocs=config.num_processes)