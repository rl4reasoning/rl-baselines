# RL Reasoning baselines

The code here is heavily inspired from the amazing repo: https://github.com/McGill-NLP/nano-aha-moment

Each baseline is implemented in a seperate hackable file.

Baselines implemented:
- Dr. GRPO
- VinePPO
- Reward Progress
- Best-of-N aware finetuning

## Installation

```bash
# create new env using conda or uv or venv
pip install -r requirements.txt
```

## Datasets
This repo was created for the blog post: ["What to do when you have zero rewards during RL?"]((https://spiffy-airbus-472.notion.site/What-Can-You-Do-When-You-Have-Zero-Rewards-During-RL-260429bdb7308024b6bdd3ed4f64c15f))

We provide data generation scripts for star-graph that was used in the above blogpost. However, one could add their own tasks as well. Have a look for [`tasks`](./tasks/) directory for inspiration.

### Star-graph dataset generation
#### Create star graph data for training
Please follow the notebook [`create_star_graph_data.ipynb`](./create_star_graph_data.ipynb) to generate a star-graph dataset and push to HF.

#### Mix Datasets
To create dataset mixtures follow instructions in [`combine_datasets.ipynb`](./combine_datasets.ipynb).

## Sample commands to run different baselines

### Dr. GRPO
```bash
python nano_r1_script.py \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--task star-graph-deg-3-path-3-nodes-300 \
--run_id Qwen2.5-1.5B-Instruct-Deg-3-Path-3
```

### VinePPO 
Chunk advantages are estimated using Monte carlo rollouts from top-3 high entropy tokens in the response
```bash
python vineppo_and_reward_progress.py \
--prover_policy_model_name Qwen/Qwen2.5-1.5B-Instruct \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--run_id custom_run_id \
--top_k_entropy_tokens 3  \
--vineppo_k 3 \
--prover_alpha 1.00 \
--prover_policy_best_of_n 1 \
--current_policy_as_prover 1 \
--task hf_username/star-graph-deg-3-path-3-nodes-300
```

### Reward Progress 
Use prover as  `Best-of-4(Qwen/Qwen2.5-1.5B-Instruct)` and advantage under the prover is estimated using roll outs from top-3 high entropy tokens
```bash
python nano_r1_script_prover.py \
--prover_policy_model_name Qwen/Qwen2.5-1.5B-Instruct \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--run_id custom_run_id \
--top_k_entropy_tokens 3 \
--vineppo_k 3 \
--prover_alpha 0.83 \
--prover_policy_best_of_n 4 \
--current_policy_as_prover 0 \
--task star-graph-deg-3-path-3-nodes-300
```

### Best-of-N aware finetuning 
Best-of-8 finetuning, using KL schedule from 0.1 to 0.001 in 1000 steps
```bash
python nano_r1_script_bon.py \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--task star-graph-deg-10-path-10-nodes-300 \
--run_id "10x10-bo8-kl-0.1-to-0.001-r2" \
--loss_type "best_of_n" \
--num_generations 8 \
--kl_schedule linear --initial_kl_coeff 0.1 --final_kl_coeff 0.001
```

## Citation
In case you find this repo helpful, consider citing it as:
```js
@misc{jpab2025rlzero,
title={What Can You Do When You Have Zero Rewards During RL?},
author={Jatin Prakash and Anirudh Buvanesh},
year={2025},
howpublished={\url{https://spiffy-airbus-472.notion.site/What-Can-You-Do-When-You-Have-Zero-Rewards-During-RL-260429bdb7308024b6bdd3ed4f64c15f}},
note={Notion Blog},
}
```