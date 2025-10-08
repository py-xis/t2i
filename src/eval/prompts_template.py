import json
import random
from itertools import chain

MARIO_CAPTION_TEMPLATES = [
    "Person with text '{text}' in hands",
    "White sign with text '{text}'",
    "Title with text '{text}'",
    "A dog with white sign saying '{text}'",
    "Text '{text}' on a board with cat",
    "A book cover with a text '{text}' on it",
]


def get_singular_texts(limit=150):
    with open("data/benchmark/laion_word/test1k.json") as file:
        data = json.load(file)
    texts = list(
        chain(
            *[
                [annotation["text"] for annotation in img["annotations"]]
                for img in data["data_list"]
            ]
        )
    )
    return random.choices(texts, k=limit)


def generate_prompt_text_dataset(limit=150):
    texts = get_singular_texts(limit=limit)
    prompts_templates = random.choices(MARIO_CAPTION_TEMPLATES, k=limit)
    prompts = [
        template.format(text=text) for template, text in zip(prompts_templates, texts)
    ]
    return [{"text": text, "prompt": prompt} for text, prompt in zip(texts, prompts)]
