import torch
import wandb
import os
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from torch.utils.data import DataLoader
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import json
import re
from drgrpo_grader import r1_zero_reward_fn
import argparse

from math_baseline import evaluate_vllm, format_prompt

from common import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)



# --- 配置参数 ---
MODEL_ID = "/home/lzc/models/Qwen2.5-Math-1.5B"
TRAIN_DATA_PATH = "/home/lzc/assignment5-alignment/data/gsm8k/train.jsonl" 
TEST_DATA_PATH = "/home/lzc/assignment5-alignment/data/gsm8k/test.jsonl" 
TRAIN_GPU = "cuda:0"
EVAL_GPU = "cuda:1"
temp_cache_dir = "/home/lzc/tmp/cache"

# 训练超参数
EPOCHS = 10
LR = 1e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32 # 模拟的总批次大小为 4 * 8 = 32
SEED = 42


'''
log_generations:
- 损失在下降，但模型只是学会了生成更短、更安全的回答（比如总是回答“我不知道”）。
- 损失在下降，但模型开始产生重复的、无意义的文本。
- 损失在下降，但模型回答的格式、语气和我们期望的完全不同。
在训练的“循环中 (in-the-loop)”定期地用模型生成一些样本，并将生成结果、评估分数等关键信息打印或记录下来，以供人工检查和分析。
'''

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.3):
    """
        Start the inference process, here we use vLLM to hold a model on
        a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    # with world_size_patch, profiling_patch:
    #     return LLM(
    #         model=model_id,
    #         # device=device,
    #         tensor_parallel_size=1,
    #         dtype=torch.bfloat16,
    #         enable_prefix_caching=True,
    #         gpu_memory_utilization=gpu_memory_utilization,
    #     )
    with patch("torch.distributed.get_world_size", return_value=1):
        return LLM(
            model=model_id,
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
        Copied from https://github.com/huggingface/trl/blob/
        22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def config_wandb():
    WANDB_PROJECT_NAME = "CS336-alignment"
    # 初始化 wandb
    wandb.init(project=WANDB_PROJECT_NAME, config=locals())
    # 设置 wandb 指标（从您提供的代码片段）
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")



def get_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))   # 或者如果是数组格式，用 json.load

    dataset = Dataset.from_list(data)
    return dataset



ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")

def extract_reasoning_and_answer(full_answer: str) -> tuple[str, str]:
    """
    将包含推理和答案的字符串分割成两部分。

    Args:
        full_answer: 包含推理过程和以 "####" 标记的答案的完整字符串。

    Returns:
        一个元组，包含两个元素：
        - reasoning (str): 推理过程部分。
        - answer (str): 提取并清洗后的数字答案。
        如果找不到答案标记，则将整个输入字符串视为 reasoning，answer 返回 "[invalid]"。
    """
    # 使用 "####" 进行分割，取第一部分作为推理过程
    # 使用 .split('####', 1) 可以确保只分割一次，更稳健
    parts = full_answer.split('####', 1)
    
    if len(parts) == 2:
        reasoning = parts[0].strip()
        # 对第二部分（可能包含答案）使用正则表达式进行精确匹配
        match = ANS_RE.search(full_answer)
        if match:
            # 提取并清理答案
            answer = match.group(1).strip().replace(",", "")
            return reasoning, answer

    # 如果没有找到 "####" 或正则表达式不匹配，则返回整个字符串和无效标记
    return full_answer.strip(), "[invalid]"




def r1_zero_template(
    prompts: list[str],
    outputs: list[str]
) -> tuple[list[str], list[str]]:
    SYSTEM_MESSAGE = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    )
    formatted_prompts = []
    formatted_responses = []

    for prompt, output in zip(prompts, outputs):
        formatted_prompt = f"{SYSTEM_MESSAGE}\nUser: {prompt}\nAssistant: <think>"

        # 找到 output 中的答案和推理过程
        reasoning, answer = extract_reasoning_and_answer(output)
        formatted_response = f"{reasoning}</think> <answer>{answer}</answer>"  # </think> <answer> 之间的空格很重要，r1_zero_reward_fn格式

        formatted_prompts.append(formatted_prompt)
        formatted_responses.append(formatted_response)

    return (formatted_prompts, formatted_responses)
    



def main():
    # config_wandb()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_size",
        type=int,
        default=-1,
        help="Use a random subset of the training data. If not specified or -1, use the full dataset."
    )
    args = parser.parse_args()


    # 加载主模型（策略模型）和分词器到训练GPU
    print(f"Loading policy model and tokenizer to {TRAIN_GPU}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(TRAIN_GPU)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(policy_model.parameters(), lr=LR)

    # 初始化 vLLM 实例到评估GPU
    print(f"Initializing vLLM on {EVAL_GPU}...")
    llm_eval_instance = init_vllm(MODEL_ID, device=EVAL_GPU, seed=SEED)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"] # 关键：设置停止符
    )

    train_dataset = get_dataset(TRAIN_DATA_PATH)
    test_dataset = get_dataset(TEST_DATA_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    train_size = args.train_size
    if train_size > 0:
        # 确保子集大小不超过原始数据集大小
        size = min(train_size, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=42).select(range(size))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    train_step = 0
    step_loss = 0

    print("\n--- Starting Training Loop ---")

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        policy_model.train()

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            train_step += 1      # mini batch

            # 需要对 prompt 和 answer 加上模板
            prompts, responses = r1_zero_template(batch['question'], batch['answer'])

            batch_dict = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = batch_dict['input_ids'].to(TRAIN_GPU)
            labels = batch_dict['labels'].to(TRAIN_GPU)
            response_mask = batch_dict['response_mask'].to(TRAIN_GPU)             # [batch_size, seq_len]

            log_probs_dict = get_response_log_probs(policy_model, input_ids, labels, False)  # [batch_size, seq_len]   [4,276]
            log_probs = log_probs_dict['log_probs']

            scale_loss, _ = sft_microbatch_train_step(log_probs, response_mask, GRADIENT_ACCUMULATION_STEPS, 1.0)
            step_loss += scale_loss

            if train_step == GRADIENT_ACCUMULATION_STEPS:
                optimizer.step()
                optimizer.zero_grad()
                step_loss /= GRADIENT_ACCUMULATION_STEPS
                # tqdm.write(f"TRAIN LOSS: {step_loss.item()}")

                train_step = 0
                step_loss = 0

        policy_model.save_pretrained("/home/lzc/model.pth")

        policy_model.eval() # 切换到评估模式
        load_policy_into_vllm_instance(policy_model, llm_eval_instance)



        test_prompts = [format_prompt(item['question']) for item in test_dataset]
        test_answers = [item['answer'] for item in test_dataset]

        evaluate_vllm(llm_eval_instance, r1_zero_reward_fn, test_prompts, test_answers, sampling_params)
        


# 
# 1. 样本数 {128, 256, 512, 1024}, along with using the full dataset. 
# 2. wandb 追踪
# 3. log_generations


if __name__ == "__main__":
    main()