import re
import json
from typing import Callable, List, Dict
from datasets import load_dataset
from tqdm import tqdm

from drgrpo_grader import r1_zero_reward_fn
from sft_train import extract_reasoning_and_answer

response = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.\n#### 18"


output = "......</think> <answer>18</answer>"

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")

def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


reasoning, answer = extract_reasoning_and_answer(response)
print(reasoning)
print(answer)

dict = r1_zero_reward_fn(output, answer)

print(dict)