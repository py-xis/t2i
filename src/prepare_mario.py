import os
import numpy as np


def extract_text_inside_quotes(s):
    start = s.find('"') + 1
    end = s.find('"', start)
    return s[start:end]


def prepare_prompts_mario_bechmark(data_file: str) -> tuple[list[dict], list[dict]]:
    prompts = []
    with open(data_file, "r") as promptf:
        for line in promptf:
            line = line.strip()
            prompts.append(line)

    prompts_A = []
    prompts_B = []

    for i, prompt in enumerate(prompts):
        text_A = extract_text_inside_quotes(prompt)
        prompts_A.append(
            {
                "text": text_A,
                "prompt": prompt,
            }
        )
        target_idx = i
        while target_idx == i:
            target_idx = np.random.randint(len(prompts))
        text_B = extract_text_inside_quotes(prompts[target_idx])
        prompts_B.append({"text": text_B, "prompt": prompt.replace(text_A, text_B)})

    assert len(prompts_A) == len(prompts_B)
    return prompts_A, prompts_B
