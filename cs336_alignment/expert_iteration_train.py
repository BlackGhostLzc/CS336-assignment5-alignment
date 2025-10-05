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

from math_baseline import evaluate_vllm, format_prompt
from sft_train import load_policy_into_vllm_instance, init_vllm

from common import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)



# 模型和路径配置
MODEL_ID = "/home/lzc/models/Qwen2.5-Math-1.5B"
TRAIN_DATA_PATH = "../data/gsm8k/train.jsonl"
TEST_DATA_PATH = "../data/gsm8k/test.jsonl"

# 设备配置
TRAIN_GPU = "cuda:0"
EVAL_GPU = "cuda:1" 

# 专家迭代 (EI) 配置
N_SFT_STEPS = 100              # 总共进行多少轮专家迭代
NUM_ROLLOUTS = 1             # G, 每个问题生成多少个候选回答

# SFT 训练配置
EPOCHS_PER_SFT_STEP = 1      # 在筛选出的新数据集上训练几轮
BATCH_SIZE = 1
LR = 1e-5
GRADIENT_ACCUMULATION_STEPS = 64

# 随机种子
SEED = 42
torch.manual_seed(SEED)


# --- 数据处理函数 (Data Handling Functions) ---
ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")

def get_dataset(path: str) -> Dataset:
    """从 JSONL 文件加载数据并创建 Hugging Face Dataset 对象"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def extract_reasoning_and_answer(full_answer: str) -> tuple[str, str]:
    """将包含推理和答案的字符串分割成两部分"""
    parts = full_answer.split('####', 1)
    if len(parts) == 2:
        reasoning = parts[0].strip()
        match = ANS_RE.search(full_answer)
        if match:
            answer = match.group(1).strip().replace(",", "")
            return reasoning, answer
    return full_answer.strip(), "[invalid]"


def r1_zero_template(prompts: list[str], outputs: list[str]) -> tuple[list[str], list[str]]:
    """为 SFT 训练格式化 prompts 和 responses"""
    formatted_prompts = []
    formatted_responses = []

    for prompt, output in zip(prompts, outputs):
        formatted_prompt = format_prompt(prompt) # 使用通用的 prompt 格式化函数
        reasoning, answer = extract_reasoning_and_answer(output)
        formatted_response = f"{reasoning}</think> <answer>{answer}</answer>"
        formatted_prompts.append(formatted_prompt)
        formatted_responses.append(formatted_response)
        
    return (formatted_prompts, formatted_responses)



def main():
    print("--- (Expert Iteration Experiment Start) ---")

    # 1. 初始化模型、分词器和优化器
    print(f"Loading policy model and tokenizer to {TRAIN_GPU}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(TRAIN_GPU)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)

    # 2. 初始化 vLLM 用于快速推理
    print(f"Initializing vLLM on {EVAL_GPU}...")
    llm_eval_instance = init_vllm(MODEL_ID, device=EVAL_GPU, seed=SEED)

    # 3. 加载完整数据集
    print("Loading datasets...")
    train_dataset = get_dataset(TRAIN_DATA_PATH)
    test_dataset = get_dataset(TEST_DATA_PATH)
    
    test_prompts = [format_prompt(item['question']) for item in test_dataset]
    test_ground_truth_answers = [item['answer'] for item in test_dataset]

    # vLLM 生成配置
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        min_tokens=4, # 避免生成空的回答
        n = NUM_ROLLOUTS
    )

    # --- 专家迭代主循环 (Expert Iteration Main Loop) ---
    for sft_step in range(N_SFT_STEPS):
        print(f"\n{'='*20} Expert Iteration Step {sft_step + 1}/{N_SFT_STEPS} {'='*20}")

        policy_model.eval()

        load_policy_into_vllm_instance(policy_model, llm_eval_instance)

        # -----------------
        # 步骤 A: 数据生成 (Data Generation)
        # -----------------
        # 为每个问题创建 NUM_ROLLOUTS 个副本以进行并行生成
        generation_prompts = []
        original_questions = []
        original_answers = []
        for item in train_dataset:
            prompt = format_prompt(item['question'])
            generation_prompts.extend([prompt] * NUM_ROLLOUTS)
            original_questions.extend([item['question']] * NUM_ROLLOUTS)
            original_answers.extend([item['answer']] * NUM_ROLLOUTS)
            
        # 使用 vLLM 生成回答
        outputs = llm_eval_instance.generate(generation_prompts, sampling_params)

        # 每个回答产生了 NUM_ROLLOUTS 个答案
        generated_completions = [output.outputs[0].text.strip() + "</answer>" for output in outputs]

        # -----------------
        # 步骤 B: 数据筛选 (Data Filtering)
        # -----------------
        print("Step B: Filtering generated data using reward function...")

        correct_qa_pairs = []
        for generated_completion, original_answer, original_question in zip(generated_completions, original_answers, original_questions):
            reasoning, answer = extract_reasoning_and_answer(original_answer)
            d = r1_zero_reward_fn(generated_completion, answer)
            if d['reward'] > 0:
                correct_qa_pairs.append({
                    "question": original_question,
                    "answer": generated_completion
                })

        
        num_correct = len(correct_qa_pairs)
        print(f"Found {num_correct} correct completions out of {len(generated_completions)} total rollouts.")
        
        if num_correct == 0:
            print("Warning: No correct completions found in this step. Skipping SFT.")
            # 评估当前模型性能
            print("\n--- Evaluating current model performance on test set ---")
            policy_model.eval()
            load_policy_into_vllm_instance(policy_model, llm_eval_instance)
            evaluate_vllm(llm_eval_instance, r1_zero_reward_fn, test_prompts, test_ground_truth_answers, SAMPLING_PARAMS, f"Step {sft_step+1} (Before SFT)")
            continue

        # -----------------
        # 步骤 C: 监督微调 (Supervised Fine-Tuning)
        # -----------------
        print(f"Step C: Starting SFT on {num_correct} new correct samples...")
        sft_dataset = Dataset.from_list(correct_qa_pairs)
        sft_dataloader = DataLoader(sft_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        policy_model.train()
        train_step_counter = 0
        total_loss = 0

        for epoch in range(EPOCHS_PER_SFT_STEP):
            print(f"SFT Epoch {epoch + 1}/{EPOCHS_PER_SFT_STEP}")
            for batch in tqdm(sft_dataloader, desc=f"SFT Step {sft_step+1}, Epoch {epoch+1}"):
                
                prompts, responses = r1_zero_template(batch['question'], batch['answer'])
                
                batch_dict = tokenize_prompt_and_output(prompts, responses, tokenizer)
                input_ids = batch_dict['input_ids'].to(TRAIN_GPU)
                labels = batch_dict['labels'].to(TRAIN_GPU)
                response_mask = batch_dict['response_mask'].to(TRAIN_GPU)
                log_probs_dict = get_response_log_probs(policy_model, input_ids, labels, False)
                log_probs = log_probs_dict['log_probs']
                
                # # SFT 训练步骤
                # loss = -log_probs.sum() / response_mask.sum()
                # scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
                # scaled_loss.backward()
                # total_loss += scaled_loss.item()
                scaled_loss, _ = sft_microbatch_train_step(log_probs, response_mask, GRADIENT_ACCUMULATION_STEPS, 1.0)
                total_loss += scaled_loss.item()

                train_step_counter += 1
                if train_step_counter % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        print(f"SFT Step {sft_step + 1} finished. Average Loss: {total_loss / len(sft_dataloader)}")

        # -----------------
        # 步骤 D: 评估 (Evaluation)
        # -----------------
        print(f"\n--- Evaluating model after SFT Step {sft_step+1} on test set ---")
        policy_model.eval()
        load_policy_into_vllm_instance(policy_model, llm_eval_instance)
        evaluate_vllm(llm_eval_instance, r1_zero_reward_fn, test_prompts, test_ground_truth_answers, sampling_params)




if __name__ == "__main__":
    main()
