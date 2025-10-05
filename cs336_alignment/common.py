from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F

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
