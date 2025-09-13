templates_dict = {
    "r1": {
        "system_prompt": None,
        # DeepSeek-R1 recommends this template on HF
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B#usage-recommendations
        "prompt_template": ("{}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."),
        # "prompt_template": (
        #     "{}\n\nThink about the reasoning process in the mind and then provides an answer. "
        #     "The reasoning process is enclosed within <think> </think> tags, i.e., "
        #     "<think> reasoning process here </think>. "
        #     "Put your final answer within \\boxed{{}}."
        # ),
        "apply_tokenizer_template": True,
    },
    # "qwen_math_old": {
    #     "system_prompt": None,
    #     "prompt_template": (
    #         "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n"
    #         "{}<|im_end|>\n<|im_start|>assistant\n"
    #     ),
    #     "apply_tokenizer_template": False,
    # },
    "qwen_math": {
        "system_prompt": None,
        "prompt_template": (
            "<|im_start|>user\n" "{} Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
        ),
        "apply_tokenizer_template": False,
    },
    "qwen_base": {
        "system_prompt": None,
        "prompt_template": (
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n"
            "{}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "apply_tokenizer_template": False,
    },
}
