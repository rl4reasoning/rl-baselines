# ─── Load environment variables early ──────────────────────────────────────────
import os

import yaml

if os.path.exists("env_vars.yml"):
    with open("env_vars.yml", "r") as f:
        env_vars = yaml.safe_load(f)
    for key, value in env_vars.items():
        os.environ[key] = value

# ─── Standard library ──────────────────────────────────────────────────────────
import argparse
import logging
from pathlib import Path

# ─── Third-party libraries ────────────────────────────────────────────────────
import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from deepspeed.runtime.utils import see_memory_usage
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# ─── Local application imports ─────────────────────────────────────────────────
from tasks import get_dataset, get_reward_fn, preprocess_example
from utils import (
    dump_episodes,
    evaluate_on_test_set,
    initialize_training_process_group,
    load_model_into_vllm,
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

arg_parser = argparse.ArgumentParser(description="Train R1 model with PPO")
arg_parser.add_argument("--kl_coeff", type=float, default=0.001, help="KL coefficient for PPO")
arg_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
arg_parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B", help="Model name/path")
arg_parser.add_argument(
    "--per_device_batch_size",
    type=int,
    default=8,
    help="Per device batch size",
)
arg_parser.add_argument("--max_response_tokens", type=int, default=2048, help="Max response tokens")
arg_parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-6,
    help="Learning rate for training",
)
arg_parser.add_argument("--debug", action="store_true", help="Debug mode")
arg_parser.add_argument(
    "--algorithm",
    type=str,
    choices=["grpo", "vineppo"],
    default="grpo",
    help="Algorithm to use",
)
arg_parser.add_argument(
    "--vineppo_k",
    type=int,
    default=3,
    help="Number of MC samples to take for each response",
)
arg_parser.add_argument("--run_id", type=str, default=None, help="Run ID")
arg_parser.add_argument(
    "--nproc",
    type=int,
    default=1,
    help="Number of processes (data parallelism) to use",
)
arg_parser.add_argument("--n", type=int, default=4, help="Number of rollouts to generate per sample during training")
arg_parser.add_argument("--n_eval", type=int, default=4, help="Number of samples to evaluate on")
arg_parser.add_argument("--eval_temperature", type=float, default=0.6, help="Temperature for evaluation")
arg_parser.add_argument("--max_eval_tokens", type=int, default=2048, help="Maximum number of tokens to evaluate on")
arg_parser.add_argument("--task", default="math", help="Task to run")
arg_parser.add_argument("--output_dir", type=str, default="results/", help="Output Directory")
arg_parser.add_argument("--load_trained_checkpoint", type=str, default="", help="Path to load trained checkpoint from")
arg_parser.add_argument("--limit_eval", type=int, default=10, help="Limit number of evaluation samples")
arg_parser.add_argument("--dump_folder", type=str, default=None, help="Folder to dump episodes")
arg_parser.add_argument("--csv_file", type=str, default="results.csv", help="CSV file to save evaluation results")


def load_dataset(task: str, tokenizer: AutoTokenizer, model_name: str):
    dataset = get_dataset(task)
    # Rank 0 will preprocess the dataset first
    if dist.get_rank() != 0:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    print("The task is", task)
    train_dataset = dataset["train"].map(
        preprocess_example,
        num_proc=6,
        fn_kwargs={
            "task": task,
            "tokenizer": tokenizer,
            "model_name": model_name,
        },
        desc="Preprocessing Train dataset",
        load_from_cache_file=False,  # done to ensure that any changes to the prompt template are reflected in the dataset
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
    # Parse command line arguments
    args = arg_parser.parse_args()
    print("##############################################")
    print(args)
    print("##############################################")
    print()

    # rank = int(os.environ.get("RANK", "0"))
    nproc = int(os.environ.get("WORLD_SIZE", "1"))
    nproc = args.nproc
    initialize_training_process_group(rank, nproc, port=8237)

    # Disable logging for non-main processes to avoid duplicate logs
    if dist.get_rank() != 0:
        logger.setLevel(logging.ERROR)

    if args.debug:
        import debugpy

        debugpy.listen(5678)
        logger.info("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        logger.info("Debugger attached")

    ############################################
    # Hyperparameters
    ############################################

    EXP_DIR = Path(args.load_trained_checkpoint)

    # Model configuration
    MODEL_NAME = args.model_name

    # RL parameters
    # Total number of training iterations
    # Number of episodes to collect per iteration for training
    print(f"Number of ranks: {dist.get_world_size()}")

    GLOBAL_BATCH_SIZE = 16
    GENERATIONS_PER_SAMPLE = args.n

    NUM_PROCESSES = dist.get_world_size()

    EPISODES_PER_ITERATION = GLOBAL_BATCH_SIZE * GENERATIONS_PER_SAMPLE * NUM_PROCESSES  # Num samples * num generations * num ranks
    EPISODES_PER_ITERATION_PER_RANK = EPISODES_PER_ITERATION // dist.get_world_size()
    # Number of responses to generate for each input prompt

    # Controls how much the policy can deviate from the reference model

    # Training hyperparameters
    # Batch size for each GPU device during training
    PER_DEVICE_BATCH_SIZE = args.per_device_batch_size
    assert EPISODES_PER_ITERATION_PER_RANK % PER_DEVICE_BATCH_SIZE == 0

    # Sampling parameters
    # Maximum number of tokens to generate in each response
    MAX_RESPONSE_TOKENS = args.max_response_tokens
    # Controls randomness in generation (higher = more random)
    # Nucleus sampling parameter (1.0 = disabled)
    # TOP_P = (
    #     0.999  # to avoid sampling unused tokens absent from tokenizer see https://github.com/vllm-project/vllm/issues/13175#issuecomment-2781842571
    # )
    # # Top-k sampling parameter (-1 = disabled)
    # TOP_K = -1  # no top k
    # Number of MC samples to take for each response
    # DeepSpeed configuration
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": False},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION_PER_RANK // PER_DEVICE_BATCH_SIZE,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True,
                "fused": True,
            },
        },
    }
    # disable deepspeed chatter
    os.environ["DEEPSPEED_LOG_LEVEL"] = "error"  # or "warning"
    logging.getLogger("DeepSpeed").setLevel(logging.ERROR)

    compute_reward_fn = get_reward_fn(args.task)

    dist.barrier(device_ids=[torch.cuda.current_device()])

    # model_name_short = MODEL_NAME.split("/")[-1]
    # if args.run_id is None:
    #     RUN_NAME = (
    #         f"{model_name_short}_temp-{TEMPERATURE}_kl-{KL_COEFFICIENT}_lr-{LEARNING_RATE}_al-{args.algorithm}_task-{args.task}_loss-{LOSS_TYPE}"
    #     )
    # else:
    #     RUN_NAME = args.run_id
    # EXP_DIR = Path(args.output_dir) / RUN_NAME
    # EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    ############################################
    # Prompts and Dataset
    ############################################

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)
    _, orig_test_datasets = load_dataset(args.task, tokenizer, MODEL_NAME)

    test_datasets = {
        test_dataset_name: orig_test_dataset.shard(num_shards=dist.get_world_size(), index=dist.get_rank())
        for test_dataset_name, orig_test_dataset in orig_test_datasets.items()
    }

    for test_dataset_name, test_dataset in test_datasets.items():
        logger.info(f"Rank: {dist.get_rank()}, Test dataset {test_dataset_name} size: {len(test_dataset)}")

    ############################################
    # Initialize Models
    ############################################

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )

    see_memory_usage("Before initializing DeepSpeed engines", force=dist.get_rank() == 0)

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )

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
        gpu_memory_utilization=0.7,
        enable_prefix_caching=True,
        swap_space=4,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=max(MAX_RESPONSE_TOKENS, args.max_eval_tokens) + 1024,
        enable_sleep_mode=True,
        device=f"cuda:{torch.cuda.current_device()}",
        tensor_parallel_size=1,
    )

    see_memory_usage("After initializing inference engine", force=dist.get_rank() == 0)

    # ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    # if ckpt_path is not None:

    if len(args.load_trained_checkpoint) == 0:
        logger.info("No checkpoint provided, starting from scratch ...")
        if args.dump_folder is not None:
            args.load_trained_checkpoint = args.dump_folder
        else:
            logger.info("No dump folder provided, will not be able to save episodes.")
        # EXP_DIR = Path(args.load_trained_checkpoint)
        # EXP_DIR.mkdir(parents=True, exist_ok=True)

        # Load the model into vLLM
        load_model_into_vllm(policy_model, inference_engine)
        iteration = 0
    else:
        ckpt_path = Path(args.load_trained_checkpoint)
        logger.info(f"Resuming from checkpoint {ckpt_path} ...")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        load_model_into_vllm(policy_model, inference_engine)
        iteration = int(ckpt_path.name.split("_")[-1])

    eval_stats = {}
    for test_dataset_name, test_dataset in test_datasets.items():
        logger.info(f"Evaluating on {test_dataset_name}...")
        logger.info(f"Limiting test dataset to {args.limit_eval} ....")
        test_dataset = test_dataset.select(range(args.limit_eval))

        _eval_episodes, _eval_stats = evaluate_on_test_set(
            inference_engine=inference_engine,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            eos_token=EOS_TOKEN,
            eval_sampling_params=SamplingParams(
                temperature=args.eval_temperature,
                max_tokens=args.max_eval_tokens,
                n=args.n_eval,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
            ),
            reward_func=lambda completion, sample: compute_reward_fn(completion, sample, EOS_TOKEN),
        )
        if len(args.load_trained_checkpoint) > 0:
            _eval_episode_table = dump_episodes(
                episodes=_eval_episodes,
                episodes_stats=_eval_stats,
                exp_dir=EXP_DIR,
                tokenizer=tokenizer,
                iteration=iteration,
                is_eval=True,
                do_save=True,
                eval_dataset_name=args.task,
            )
            print(_eval_episode_table)
        for k, v in _eval_stats.items():
            # print(k, v)
            eval_stats[f"{test_dataset_name}/{k}"] = v
        print("\n################################################\n")
        print("Using task:", args.task)
        print("Using checkpoint:", args.load_trained_checkpoint if len(args.load_trained_checkpoint) > 0 else "None")
        print()
        logs = {f"eval/{k}": np.mean(v).round(2) for k, v in eval_stats.items()}
        # pretty print the logs
        # wandb.log(logs)
        # print(logs)
        for k, v in logs.items():
            print(f"{k}: {v}")
        print()

        correct_rewards = np.array(_eval_stats["reward_metrics/answer_reward"])
        # calculate pass@k

        correct_rewards = correct_rewards.reshape(-1, args.n_eval).sum(axis=1)
        pass_at_k = np.mean(correct_rewards > 0)
        print(f"Pass@{args.n_eval}: {pass_at_k:.2f}")

        print("\n################################################\n")

        # dump this to a csv file
        if args.csv_file is not None:
            logger.info(f"Saving evaluation results to {args.csv_file} ...")

            df = dict()
            df["model_name"] = [args.model_name]
            df["model_path"] = [args.load_trained_checkpoint if len(args.load_trained_checkpoint) > 0 else "None"]
            df["task"] = [args.task]
            df["rank"] = [dist.get_rank()]

            df["num_test_samples"] = [len(test_dataset)]

            df[f"pass@{args.n_eval}"] = [pass_at_k]

            for k, v in logs.items():
                if isinstance(v, float):
                    df[k] = [v]
                elif isinstance(v, np.ndarray):
                    df[k] = [v.tolist()]
                else:
                    df[k] = [v]

            df = pd.DataFrame(df)
            df.to_csv(args.csv_file, mode="a", header=not Path(args.csv_file).exists(), index=False)

    # this results in core dumped for some reason 🤷
    # dist.barrier(device_ids=[torch.cuda.current_device()])
    # dist.destroy_process_group()


if __name__ == "__main__":
    args = arg_parser.parse_args()

    n_gpus = torch.cuda.device_count()
    if args.nproc > n_gpus:
        raise ValueError(f"Requested {args.nproc} processes, but only {n_gpus} GPUs are available.")
    print(f"Using {args.nproc} GPUs")
    if args.nproc == 1:
        main(rank=0)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.nproc)