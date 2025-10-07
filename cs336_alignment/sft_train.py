import torch
import wandb
import os
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from torch.utils.data import DataLoader

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import json
import re
from drgrpo_grader import r1_zero_reward_fn
import argparse

from math_baseline import evaluate_vllm, format_prompt

from common import *


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

    


def config_wandb():
    WANDB_PROJECT_NAME = "CS336-alignment"
    # 初始化 wandb
    wandb.init(project=WANDB_PROJECT_NAME, config=locals())
    # 设置 wandb 指标（从您提供的代码片段）
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")




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