import numpy as np
import random

np.random.seed(42)

def extract_text_inside_quotes(s):
    start = s.find('"') + 1
    end = s.find('"', start)
    return s[start:end]


template = "a cat holding a board with text '<text>'"


def prepare_prompts_wiki1000(
    data_file: str, limit=None
) -> tuple[list[dict], list[dict]]:
    texts = []
    with open(data_file, "r") as promptf:
        for line in promptf:
            line = line.strip()
            texts.append(line)

    prompts_A = []
    prompts_B = []

    for i, text_A in enumerate(texts):
        prompts_A.append(
            {
                "text": text_A,
                "prompt": template.replace("<text>", text_A),
            }
        )
        target_idx = i
        while target_idx == i:
            target_idx = np.random.randint(len(texts))
        text_B = texts[target_idx]
        prompts_B.append({"text": text_B, "prompt": template.replace("<text>", text_B)})

        if limit is not None and i + 1 >= limit:
            break
    assert len(prompts_A) == len(prompts_B)
    return prompts_A, prompts_B


wiki400_templates = [
    "A sign that says '<text>'",
    "New York Skyline with '<text>' written with fireworks on the sky",
    "A storefront with '<text>' written on it",
]


def prepare_prompts_wiki400(
    data_file: str, limit=None
) -> tuple[list[dict], list[dict]]:
    texts = []
    with open(data_file, "r") as promptf:
        for line in promptf:
            line = line.strip()
            texts.append(line)

    prompts_A = []
    prompts_B = []

    for i, text_A in enumerate(texts):
        for pt in wiki400_templates:
            prompts_A.append(
                {
                    "text": text_A,
                    "prompt": pt.replace("<text>", text_A),
                }
            )
            target_idx = i
            while target_idx == i:
                target_idx = np.random.randint(len(texts))
            text_B = texts[target_idx]
            prompts_B.append({"text": text_B, "prompt": pt.replace("<text>", text_B)})

        if limit is not None and i + 1 >= limit:
            break
    assert len(prompts_A) == len(prompts_B)
    return prompts_A, prompts_B


def prepare_prompts_wiki398(
    data_file: str, limit=None
) -> tuple[list[dict], list[dict]]:
    texts = []
    with open(data_file, "r") as promptf:
        for line in promptf:
            line = line.strip()
            texts.append(line)

    length_dict = {}
    for word in texts:
        length = len(word)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(word)

    prompts_A = []
    prompts_B = []

    for i, text_A in enumerate(texts):
        for pt in wiki400_templates:
            prompts_A.append(
                {
                    "text": text_A,
                    "prompt": pt.replace("<text>", text_A),
                }
            )
            text_B = text_A
            while text_B == text_A:
                text_B = random.choice(length_dict[len(text_A)])
            prompts_B.append({"text": text_B, "prompt": pt.replace("<text>", text_B)})

        if limit is not None and i + 1 >= limit:
            break
    assert len(prompts_A) == len(prompts_B)
    return prompts_A, prompts_B
