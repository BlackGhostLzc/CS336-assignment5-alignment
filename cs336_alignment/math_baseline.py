import re
import json
from typing import Callable, List, Dict
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

from drgrpo_grader import r1_zero_reward_fn



ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")

def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


def format_prompt(question: str) -> str:
    # 根据 r1_zero 的要求格式化问题为提示
    return (f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
            f"User: {question}"
            f"Assistant: <think>"
    )



def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    print(f"Begin evaluate {len(prompts)} samples...")

    # 1. 使用vllm批量生成所有prompts的输出
    model_outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    total_correct = 0
    total_format_correct = 0
    total_answer_correct = 0
    
    outputs = []

    # 2. 遍历每个生成结果进行评估
    for i, output in enumerate(tqdm(model_outputs, desc="Evaluating")):
        prompt_text = output.prompt
        generated_text = output.outputs[0].text.strip()
        ground_truth_answer = answers[i]
        ground_truth_answer = extract_reference_answer(ground_truth_answer)

        outputs.append(generated_text)
        
        # 3. 使用奖励函数计算分数
        score_dict = reward_fn(generated_text, ground_truth_answer)
        
        # 'reward' == 1.0 表示格式和答案都正确
        if score_dict.get('reward', 0.0) == 1.0:
            total_correct += 1
        
        # 'format_reward' == 1.0 表示格式正确
        if score_dict.get('format_reward', 0.0) == 1.0:
            total_format_correct += 1

        # 'answer_reward' == 1.0 表示答案正确 (通常在格式也正确时才有意义)
        if score_dict.get('answer_reward', 0.0) == 1.0:
            total_answer_correct += 1
            
        # 4. 将该样本的详细信息存入结果列表
        results.append({
            "prompt": prompt_text,
            "question": prompts[i], # 原始问题
            "generated_text": generated_text,
            "ground_truth": ground_truth_answer,
            "scores": score_dict
        })

        
    # 5. 计算整体评估指标（准确率）
    reward_accuracy = total_correct / len(prompts) if prompts else 0.0
    format_accuracy = total_format_correct / len(prompts) if prompts else 0.0
    answer_accuracy = total_answer_correct / len(prompts) if prompts else 0.0

    print("\nEvaluate end")
    print(f"格式和答案都正确率: {reward_accuracy:.4f} ({total_correct}/{len(prompts)})")
    print(f"格式正确率: {format_accuracy:.4f} ({total_format_correct}/{len(prompts)})")
    print(f"答案正确率: {answer_accuracy:.4f} ({total_answer_correct}/{len(prompts)})")


if __name__ == "__main__":
    MODEL_PATH = "/home/lzc/models/Qwen2.5-Math-1.5B" # 修改为你的模型路径
    DATASET_PATH = "../data/gsm8k/test.jsonl"  # 修改为你的 test.jsonl 文件路径
    OUTPUT_FILE = "../data/gsm8k/gsm8k_qwen1.5b_zeroshot_results.json"

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"] # 关键：设置停止符
    )

    llm = LLM(model=MODEL_PATH, trust_remote_code=True, enforce_eager=True)

    dataset = load_dataset("json", data_files=DATASET_PATH)
    dataset = dataset['train']
   
    prompts = [format_prompt(item['question']) for item in dataset]
    answers = [item['answer'] for item in dataset]


    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn, # 传入为GSM8K定制的奖励函数
        prompts=prompts,
        answers=answers,
        eval_sampling_params=sampling_params,
    )
