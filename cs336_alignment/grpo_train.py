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
from argparse import Namespace
import itertools
from math_baseline import evaluate_vllm, format_prompt

from common import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="GRPO configs")

    parser.add_argument("--TRAIN_GPU", type=str, default="cuda:0", help="Train GPU")
    parser.add_argument("--EVAL_GPU", type=str, default="cuda:1", help="EVAL GPU")
    parser.add_argument("--ROLLOUT_GPU", type=str, default="cuda:2", help="EVAL GPU")

    parser.add_argument("--TRAIN_DATA_PATH", type=str, default="/home/lzc/assignment5-alignment/data/gsm8k/train.jsonl")
    parser.add_argument("--TEST_DATA_PATH", type=str, default="/home/lzc/assignment5-alignment/data/gsm8k/test.jsonl")

    parser.add_argument("--MODEL_ID", type=str, default="/home/lzc/models/Qwen2.5-Math-1.5B", help="model path")

    parser.add_argument("--SEED", type=int, default=42, help="seed")

    args = parser.parse_args()
    return args



args = parse_arguments()


def grpo_train_loop(
    policy_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    llm_rollout_instance: LLM,
    train_dataset: Dataset,
    test_dataset: Dataset,
    # --- Hyperparameters ---
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    group_size: int = 4,
    rollout_batch_size: int = 256,                # 更新 1 次权重的 batch 大小, == group_size * prompts_number
    sampling_temperature: float = 1.0,        
    sampling_min_tokens: int = 4,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    loss_type: str = "grpo_clip",
    cliprange: float = 0.2, # Required for GRPO-Clip
    validation_interval: int = 10,
    update_old_model_interval: int=40,
    grad_clip_value: float = 1.0,
    use_std_normalization: bool = True,
):
    """
    Main training loop for GRPO.
    """ 
    assert train_batch_size == rollout_batch_size, "train_batch_size must equal rollout_batch_size for on-policy training"
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"


    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    micro_batch_size = train_batch_size // gradient_accumulation_steps

    old_model = AutoModelForCausalLM.from_pretrained(args.MODEL_ID, torch_dtype=torch.bfloat16).to(args.TRAIN_GPU)

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=1024, # A reasonable max length
        stop=["</answer>"]
    )

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=n_prompts_per_rollout_batch, shuffle=True)
    train_iter = itertools.cycle(train_dataloader)

    for step in tqdm(range(n_grpo_steps), desc="GRPO Steps"):
        policy_model.train()

        batch = next(train_iter)
        prompts = [format_prompt(q) for q in batch['question']]
        ground_truths = batch['answer']            
        ground_answers = []
        for answer in ground_truths:
            # 要从 ground truth 中提取出最后的答案才好进行打分
            _, format_answer = extract_reasoning_and_answer(answer)
            ground_answers.append(format_answer)

        prompts_for_rollout = [p for p in prompts for _ in range(group_size)]
        repeated_ground_answers = [gt for gt in ground_answers for _ in range(group_size)]

        outputs = llm_rollout_instance.generate(prompts_for_rollout, sampling_params)
        rollout_responses = [output.outputs[0].text + "</answer>" for output in outputs]

        
        # 要对 rollout_responses 进行打分，根据 ground truth
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(r1_zero_reward_fn, \
                        rollout_responses, repeated_ground_answers, group_size, advantage_eps, use_std_normalization)
        advantages = advantages.to(args.TRAIN_GPU)
        
        batch_dict = tokenize_prompt_and_output(prompts_for_rollout, rollout_responses, tokenizer)
        input_ids = batch_dict['input_ids'].to(args.TRAIN_GPU)
        labels = batch_dict['labels'].to(args.TRAIN_GPU)
        response_mask = batch_dict['response_mask'].to(args.TRAIN_GPU)             # [batch_size, seq_len]

        # Training Step (with Gradient Accumulation) ---
        total_loss = 0
        for i in range(0, rollout_batch_size, micro_batch_size):
            start, end = i, i + micro_batch_size

            # Slice the batch for gradient accumulation
            micro_input_ids = input_ids[start:end]
            micro_labels = labels[start:end]
            micro_response_mask = response_mask[start:end]
            micro_advantages = advantages[start:end]

            # forward pass
            policy_log_probs_dict = get_response_log_probs(policy_model, micro_input_ids, micro_labels, False)  # [batch_size, seq_len]
            policy_log_probs = policy_log_probs_dict['log_probs']

            # 获取 old model 的概率
            # old_model = llm_rollout_instance.llm_engine.model_executor.driver_worker.model_runner.model
            old_log_probs_dict = get_response_log_probs(old_model, micro_input_ids, micro_labels, False)
            old_log_probs = old_log_probs_dict["log_probs"].detach()

            scaled_loss, _ = grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, \
                                                        loss_type, raw_rewards, advantages, old_log_probs, cliprange)
            total_loss += scaled_loss
        

        total_loss /= gradient_accumulation_steps
        print("TRAIN LOSS: ", total_loss)

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip_value)
        optimizer.step()
        optimizer.zero_grad()

        if step > 0 and step % validation_interval == 0:
            policy_model.eval()
            eval_model_path = "/home/lzc/models/Alignment-Qwen2.5-Math-1.5B"
            policy_model.save_pretrained(eval_model_path)

            test_prompts = [format_prompt(item['question']) for item in test_dataset]
            test_answers = [item['answer'] for item in test_dataset]

            llm_eval_instance = LLM(
                model=eval_model_path,
                device=args.EVAL_GPU, # Use the EVAL_GPU
                seed=args.SEED,
                trust_remote_code=True
            )

            evaluate_vllm(llm_eval_instance, r1_zero_reward_fn, test_prompts, test_answers, sampling_params)

        if step > 0 and step % update_old_model_interval == 0:
            eval_model_path = "/home/lzc/models/Alignment-Qwen2.5-Math-1.5B"
            policy_model.save_pretrained(eval_model_path)

            llm_rollout_instance = init_vllm(eval_model_path, device=args.ROLLOUT_GPU, seed=args.SEED)
            old_model = AutoModelForCausalLM.from_pretrained(eval_model_path, torch_dtype=torch.bfloat16).to(args.TRAIN_GPU)




def main():
    print(f"Loading policy model and tokenizer to {args.TRAIN_GPU}...")
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_model = AutoModelForCausalLM.from_pretrained(args.MODEL_ID, torch_dtype=torch.bfloat16).to(args.TRAIN_GPU)


    # 这个 llm_rollout_instance 是 old model
    print(f"Initializing vLLM on {args.ROLLOUT_GPU}...")
    llm_rollout_instance = init_vllm(args.MODEL_ID, device=args.ROLLOUT_GPU, seed=args.SEED)

    train_dataset = get_dataset(args.TRAIN_DATA_PATH)
    test_dataset = get_dataset(args.TEST_DATA_PATH)
    
    print("Starting GRPO training loop...")
    grpo_train_loop(
        policy_model=policy_model,
        tokenizer=tokenizer,
        llm_rollout_instance=llm_rollout_instance,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )



if __name__ == "__main__":
    main()