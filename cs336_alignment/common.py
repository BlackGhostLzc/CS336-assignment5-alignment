from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import torch.nn.functional as F

import json
from unittest.mock import patch

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

import re

########################################### SFT ################################################

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    batch_size = len(prompt_strs)
    '''
        [1, 2, 3, 4, 5, 6, 7 | 8, 9, 10, 11, 12, 13, pad, pad]
        prompt                output
        input_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, pad]
        labels:    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, pad, pad]
        mask:      [0, 0, 0, 0, 0, 0, 1, 1, 1,  1,  1,  1,  0,   0]
    '''

    # 分别对 prompt 和 output 进行分词 (不添加特殊字符，不padding)
    prompt_tokenized = tokenizer(prompt_strs, add_special_tokens=False)
    output_tokenized = tokenizer(output_strs, add_special_tokens=False)

    # 准备拼接后的序列和掩码
    all_full_ids = []
    prompt_len = []
    output_len = []
    # 2. 遍历每个样本，手动拼接
    '''
        这里不能先拼接prompt和output后再进行tokenization,这样会有一些问题，比如说
        prompt的末尾和output的开始会进行分词合并,从而导致序列变短。
    '''
    for i in range(len(prompt_strs)):
        prompt_ids = prompt_tokenized.input_ids[i]
        output_ids = output_tokenized.input_ids[i]
        full_ids = prompt_ids + output_ids
        all_full_ids.append(full_ids)
        
        prompt_len.append(len(prompt_ids))
        output_len.append(len(output_ids))


    max_len = max(len(ids) for ids in all_full_ids)

    # 准备一个新列表，用于存放填充后的结果
    padded_sequences = []
    for ids in all_full_ids:
        padding_len = max_len - len(ids)
        padding_tokens = [tokenizer.pad_token_id] * padding_len
        padded_seq = ids + padding_tokens
        padded_sequences.append(padded_seq)

    # 准备一个新列表，用于存放mask
    masks = []
    for i in range(batch_size):
        masks.append([False]*(prompt_len[i]-1) + [True]*(output_len[i]) + \
                     [False]*(max_len - prompt_len[i] - output_len[i]))

        
    final_padded_tensor = torch.tensor(padded_sequences, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.bool)

    return {
        "input_ids": final_padded_tensor[:, :-1],
        "labels": final_padded_tensor[:, 1:],
        "response_mask":masks,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    '''
        logits:  [batch_size, seq_len, vocab_size]
        returns: [batch_size, sequence_length). The entropy for each next-token prediction.     
    '''
    # 数值稳定性， log-sum-exp 技巧
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = probs * log_probs

    entropy = -torch.sum(p_log_p, dim=-1)
    return entropy



def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    # 1. 首先用 input_ids 输入给模型做一次前向计算 logits
    logits = model(input_ids).logits       # [batch_size, seq_len, vocab_size]

    log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
    labels_for_gather = labels.unsqueeze(-1)   # [batch_size, seq_len, 1]
    log_probs = log_probs.gather(dim=-1, index=labels_for_gather) # [batch_size, seq_len, 1]
    log_probs = log_probs.squeeze(-1)

    if return_token_entropy == True:
        entropy = compute_entropy(logits)
        return {
            "log_probs": log_probs,
            "token_entropy": entropy
        }
    
    else:
        return {
            "log_probs": log_probs
        }



def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    # SFT的训练目标是最小化负对数似然损失
    masked_tensor = tensor * mask

    summed_value = torch.sum(masked_tensor, dim=dim)
    '''
        一个很长的句子和一个很短的句子,即使模型在每个词元上的预测表现完全一样，它们的总损失也会相差巨大。
    '''
    normalized_value = summed_value / normalize_constant

    return normalized_value




def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    '''
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
            SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.

        Returns:
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.
            metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    '''
    # 后面的 loss 需要除以 batch_size
    batch_size = policy_log_probs.shape[0]

    loss = masked_normalize(
        tensor = -policy_log_probs,
        mask = response_mask,
        normalize_constant = normalize_constant,
        dim = None  # 对所有被选中的元素求和，得到一个标量
    )

    loss = loss / batch_size
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    metadata = {'loss': scaled_loss.detach()}

    return scaled_loss.detach(), metadata








########################################### GRPO ################################################

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    total_len = len(rollout_responses)
    groups_num = total_len // group_size

    raw_rewards = []
    for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        d = reward_fn(rollout_response, ground_truth)
        raw_rewards.append(d['reward'])

    raw_rewards = torch.tensor(raw_rewards)
    raw_rewards = raw_rewards.view(-1, group_size)      # [num_group, group_size]
    group_mean = torch.mean(raw_rewards, dim=1, keepdim=True)   # [num_group, 1]

    if normalize_by_std:
        group_std = torch.std(raw_rewards, dim=1, keepdim=True)
        advantages = (raw_rewards - group_mean) / (group_std + advantage_eps)
    else:
        # If False, just subtract the mean (centering).
        advantages = raw_rewards - group_mean

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
    }

    return advantages.view(-1), raw_rewards.view(-1), metadata




def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -raw_rewards_or_advantages * policy_log_probs



def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).

    当 Advantage > 0 (结果是好的): 我们希望增大 ratio 来鼓励这个行为。但 min 函数会选择 ratio 和 clipped_ratio 中较小的一个，
    限制你的“奖励”，防止你过于“贪心”而迈出危险的一步。


    """
    # 1. 从对数概率中还原真实的概率
    # ratio = pi_theta / pi_theta_old = exp(log_pi_theta - log_pi_theta_old)
    # The detach() is important to prevent gradients from flowing back to the old policy.
    log_ratios = policy_log_probs - old_log_probs.detach()  # (batch_size, sequence_length)
    ratios = torch.exp(log_ratios) # (batch_size, sequence_length)

    # 2. 计算 unclip loss
    unclipped_loss = advantages * ratios

    # 3. 对 loss 进行 clip
    clipped_ratio = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    clipped_loss = clipped_ratio * advantages

    # 4. 还要取 unclip loss 和 clip loss 的最小值
    loss = -torch.min(unclipped_loss, clipped_loss)

    # 5. Create metadata to track which tokens were clipped.
    # A token is clipped if its ratio was outside the allowed range.
    clipped = (ratios < 1.0 - cliprange) | (ratios > 1.0 + cliprange)
    metadata = {"clipped": clipped}

    return loss, metadata

    

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    # 直接使用原始的、未经处理的奖励 raw_rewards 作为优势 Advantage。
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), None
    
    # 不再使用原始奖励，而是使用经过归一化处理后的奖励，也就是 advantages。
    elif loss_type == "reinforce_with_baseline":
        loss = -policy_log_probs * advantages
        return loss, None
    
    # Loss = -min(ratio * A, clip(ratio) * A)
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)



def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    masked_tensor = tensor * mask
    numerator = masked_tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)

    return numerator / mask_sum



def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    gradient_loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, \
                                                    advantages, old_log_probs, cliprange)
    masked_mean_loss = masked_mean(gradient_loss, response_mask, None)

    scaled_loss = masked_mean_loss / gradient_accumulation_steps

    scaled_loss.backward()

    return scaled_loss.detach(), metadata








######################################################### utils ######################################################

def get_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))   # 或者如果是数组格式，用 json.load

    dataset = Dataset.from_list(data)
    return dataset



def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.2):
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
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            # device=device,
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
