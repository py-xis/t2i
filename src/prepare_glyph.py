import os
import random

import numpy as np

np.random.seed(42)
random.seed(42)

SIMPLE_PATH = "GlyphControl-release/text_prompts/raw/SimpleBench"
CREATIVE_PROMPT_TEMPLATES_PATH = (
    "GlyphControl-release/text_prompts/raw/CreativeBench/GlyphDraw_origin_remove_render_words.txt"
)
TOXIC_PATH = "data/offensive_words.txt"


template = 'A sign that says "<text>".'


def prepare_prompts_glyph_simple_bench(n_samples_per_prompt=1):
    prompts_A = []
    prompts_B = []
    prompts_AB = []
    templates_A = []
    templates_B = []

    word_files = os.listdir(SIMPLE_PATH)
    word_files = [f for f in word_files if "all_unigram_top_1000_100.txt" not in f]

    for word_file in word_files:
        with open(os.path.join(SIMPLE_PATH, word_file), "r") as promptf:
            all_words = promptf.readlines()
            text_B_indices = set(range(len(all_words)))
            for i, text_A in enumerate(all_words):
                text_B_ind = np.random.choice(list(text_B_indices))
                while text_B_ind == i:
                    text_B_ind = np.random.choice(list(text_B_indices))
                text_B = all_words[text_B_ind]
                for _ in range(n_samples_per_prompt):
                    prompts_A.append(
                        {
                            "text": text_A.strip(),
                            "prompt": template.replace("<text>", text_A.strip()),
                        }
                    )
                    prompts_B.append(
                        {
                            "text": text_B.strip(),
                            "prompt": template.replace("<text>", text_B.strip()),
                        }
                    )
                    prompts_AB.append(
                        {
                            "text": text_B.strip(),
                            "prompt": template.replace('""', f'"{text_B.strip()}"'),
                        }
                    )
                    templates_A.append(template)
                    templates_B.append(template)
                text_B_indices.remove(text_B_ind)
    assert len(prompts_A) == len(prompts_B)
    assert len(prompts_A) == 300 * n_samples_per_prompt
    assert len(prompts_B) == 300 * n_samples_per_prompt
    assert set([p["text"] for p in prompts_A]) == set([p["text"] for p in prompts_B])
    return prompts_A, prompts_B, prompts_AB, templates_A, templates_B


def prepare_prompts_glyph_simple_bench_top_100(n_samples_per_prompt=1):
    prompts_A = []
    prompts_B = []

    with open(os.path.join(SIMPLE_PATH, "all_unigram_top_1000_100.txt"), "r") as promptf:
        all_words = promptf.readlines()
        text_B_indices = set(range(len(all_words)))
        for i, text_A in enumerate(all_words):
            text_B_ind = np.random.choice(list(text_B_indices))
            while text_B_ind == i:
                text_B_ind = np.random.choice(list(text_B_indices))
            text_B = all_words[text_B_ind]
            for _ in range(n_samples_per_prompt):
                prompts_A.append(
                    {
                        "text": text_A.strip(),
                        "prompt": template.replace("<text>", text_A.strip()),
                    }
                )
                prompts_B.append(
                    {
                        "text": text_B.strip(),
                        "prompt": template.replace("<text>", text_B.strip()),
                    }
                )
            text_B_indices.remove(text_B_ind)
    assert len(prompts_A) == len(prompts_B)
    assert len(prompts_A) == 100 * n_samples_per_prompt
    assert len(prompts_B) == 100 * n_samples_per_prompt
    assert set([p["text"] for p in prompts_A]) == set([p["text"] for p in prompts_B])
    return prompts_A, prompts_B

# def prepare_prompts_glyph_creative_bench_top_100(n_samples_per_prompt=1, use_different_templates=False):
#     # Same return signature as prepare_prompts_glyph_creative_bench
#     prompts_A, prompts_B, prompts_AB, templates_A, templates_B = [], [], [], [], []

#     with open(CREATIVE_PROMPT_TEMPLATES_PATH, "r") as promptf:
#         prompt_templates = promptf.readlines()

#     # Only the top-100 list
#     top100_path = os.path.join(SIMPLE_PATH, "all_unigram_top_1000_100.txt")
#     with open(top100_path, "r") as promptf:
#         all_words = promptf.readlines()

#     text_B_indices_init = set(range(len(all_words)))
#     for i, text_A in enumerate(all_words):
#         # fresh pool each outer iteration to avoid running out too early
#         text_B_indices = set(text_B_indices_init)
#         if i in text_B_indices:
#             text_B_indices.remove(i)
#         if not text_B_indices:
#             continue
#         text_B_ind = np.random.choice(list(text_B_indices))
#         text_B = all_words[text_B_ind]

#         t1 = np.random.choice(prompt_templates).strip()
#         if use_different_templates:
#             t2 = np.random.choice(prompt_templates).strip()
#             while t2 == t1:
#                 t2 = np.random.choice(prompt_templates).strip()
#         else:
#             t2 = t1

#         for _ in range(n_samples_per_prompt):
#             prompts_A.append({"text": text_A.strip(), "prompt": t1.replace('""', f'"{text_A.strip()}"')})
#             prompts_B.append({"text": text_B.strip(), "prompt": t2.replace('""', f'"{text_B.strip()}"')})
#             prompts_AB.append({"text": text_B.strip(), "prompt": t1.replace('""', f'"{text_B.strip()}"')})
#             templates_A.append(t1)
#             templates_B.append(t2)

#     assert len(prompts_A) == len(prompts_B)
#     # top-100 list → 100 * n_samples_per_prompt
#     assert len(prompts_A) == 100 * n_samples_per_prompt
#     assert len(prompts_B) == 100 * n_samples_per_prompt
#     if use_different_templates:
#         assert len(prompts_AB) == 100 * n_samples_per_prompt
#         assert set([p["text"] for p in prompts_B]) == set([p["text"] for p in prompts_AB])
#     assert set([p["text"] for p in prompts_A]) == set([p["text"] for p in prompts_B])
#     return prompts_A, prompts_B, prompts_AB, templates_A, templates_B


def prepare_prompts_glyph_creative_bench_top_100(n_samples_per_prompt=1, use_different_templates=False, np_seed=42):
    import numpy as np, os, random
    np.random.seed(np_seed); random.seed(np_seed)

    prompts_A, prompts_B, prompts_AB = [], [], []
    templates_A, templates_B = [], []

    # load templates
    with open(CREATIVE_PROMPT_TEMPLATES_PATH, "r") as f:
        prompt_templates = [t.strip() for t in f.readlines() if t.strip()]

    # load exactly the top-100 words
    with open(os.path.join(SIMPLE_PATH, "all_unigram_top_1000_100.txt"), "r") as f:
        all_words = [w.strip() for w in f.readlines() if w.strip()]
    n = len(all_words)  # should be 100

    # make a derangement so each i maps to j != i, covering all words once
    idx = np.arange(n)
    perm = np.random.permutation(n)
    fixed = np.where(perm == idx)[0]
    if fixed.size == n:
        # rare case: identity permutation; simple rotation deranges it
        perm = np.roll(perm, 1)
    else:
        for i in fixed:
            # pick a j != i and swap; this resolves the fixed point(s) without creating new ones
            j = np.random.randint(0, n - 1)
            if j >= i:
                j += 1
            perm[i], perm[j] = perm[j], perm[i]
    assert not np.any(perm == idx)

    for i in range(n):
        text_A = all_words[i]
        text_B = all_words[perm[i]]

        tA = random.choice(prompt_templates)
        if use_different_templates:
            tB = random.choice(prompt_templates)
            # ensure different template if requested
            while tB == tA and len(prompt_templates) > 1:
                tB = random.choice(prompt_templates)
        else:
            tB = tA

        for _ in range(n_samples_per_prompt):
            prompts_A.append({"text": text_A, "prompt": tA.replace('""', f'"{text_A}"')})
            prompts_B.append({"text": text_B, "prompt": tB.replace('""', f'"{text_B}"')})
            prompts_AB.append({"text": text_B, "prompt": tA.replace('""', f'"{text_B}"')})
            templates_A.append(tA)
            templates_B.append(tB)

    # these will now hold
    assert len(prompts_A) == len(prompts_B) == 100 * n_samples_per_prompt
    assert set(p["text"] for p in prompts_A) == set(p["text"] for p in prompts_B)

    if use_different_templates:
        assert len(prompts_AB) == 100 * n_samples_per_prompt
        assert set(p["text"] for p in prompts_B) == set(p["text"] for p in prompts_AB)

    return prompts_A, prompts_B, prompts_AB, templates_A, templates_B

def prepare_prompts_glyph_creative_bench(n_samples_per_prompt=1, use_different_templates=False):
    prompts_A = []
    prompts_B = []
    prompts_AB = []
    templates_A = []
    templates_B = []
    with open(CREATIVE_PROMPT_TEMPLATES_PATH, "r") as promptf:
        prompt_templates = promptf.readlines()

    word_files = os.listdir(SIMPLE_PATH)
    word_files = [f for f in word_files if "all_unigram_top_1000_100.txt" not in f]

    for word_file in word_files:
        with open(os.path.join(SIMPLE_PATH, word_file), "r") as promptf:
            all_words = promptf.readlines()
            text_B_indices = set(range(len(all_words)))
            for i, text_A in enumerate(all_words):
                text_B_ind = np.random.choice(list(text_B_indices))
                while text_B_ind == i:
                    text_B_ind = np.random.choice(list(text_B_indices))
                text_B = all_words[text_B_ind]
                random_template = np.random.choice(prompt_templates).strip()
                if use_different_templates:
                    random_template_2 = np.random.choice(prompt_templates).strip()
                    while random_template_2 == random_template:
                        random_template_2 = np.random.choice(prompt_templates).strip()
                else:
                    random_template_2 = random_template
                for _ in range(n_samples_per_prompt):
                    prompts_A.append(
                        {
                            "text": text_A.strip(),
                            "prompt": random_template.replace('""', f'"{text_A.strip()}"'),
                        }
                    )
                    prompts_B.append(
                        {
                            "text": text_B.strip(),
                            "prompt": random_template_2.replace('""', f'"{text_B.strip()}"'),
                        }
                    )
                    prompts_AB.append(
                        {
                            "text": text_B.strip(),
                            "prompt": random_template.replace('""', f'"{text_B.strip()}"'),
                        }
                    )
                    templates_A.append(random_template)
                    templates_B.append(random_template_2)
                text_B_indices.remove(text_B_ind)

    assert len(prompts_A) == len(prompts_B)
    assert len(prompts_A) == 300 * n_samples_per_prompt
    assert len(prompts_B) == 300 * n_samples_per_prompt
    if use_different_templates:
        assert len(prompts_AB) == 300 * n_samples_per_prompt
        assert set([p["text"] for p in prompts_B]) == set([p["text"] for p in prompts_AB])
    assert set([p["text"] for p in prompts_A]) == set([p["text"] for p in prompts_B])
    return prompts_A, prompts_B, prompts_AB, templates_A, templates_B


def prepare_toxic_bench(n_samples_per_prompt=1):
    template = 'A sign that says "<text>".'
    prompts_A = []
    prompts_B = []

    word_files = os.listdir(SIMPLE_PATH)
    all_nice_words = []
    for word_file in word_files:
        with open(os.path.join(SIMPLE_PATH, word_file), "r") as promptf:
            all_nice_words.extend(promptf.readlines())

    with open(TOXIC_PATH, "r") as promptf:
        all_words = promptf.readlines()
        text_B_indices = set(range(len(all_words)))
        for i, text_A in enumerate(all_words):
            text_B_ind = np.random.choice(list(text_B_indices))
            while text_B_ind == i:
                text_B_ind = np.random.choice(list(text_B_indices))
            text_B = all_nice_words[text_B_ind]
            for _ in range(n_samples_per_prompt):
                prompts_A.append(
                    {
                        "text": text_A.strip(),
                        "prompt": template.replace("<text>", text_A.strip()),
                    }
                )
                prompts_B.append(
                    {
                        "text": text_B.strip(),
                        "prompt": template.replace("<text>", text_B.strip()),
                    }
                )
            text_B_indices.remove(text_B_ind)
    assert len(prompts_A) == len(prompts_B)
    assert len(prompts_A) == 400 * n_samples_per_prompt
    assert len(prompts_B) == 400 * n_samples_per_prompt
    return prompts_A, prompts_B


def prepare_prompts_glyph_creative_text_len(n_samples_per_prompt: int = 1, text_len: int = 1, np_seed: int = 0):
    np.random.seed(np_seed)
    random.seed(np_seed)
    prompts_A = []
    prompts_B = []
    prompts_AB = []
    templates_A = []
    templates_B = []
    with open(CREATIVE_PROMPT_TEMPLATES_PATH, "r") as promptf:
        prompt_templates = promptf.readlines()

    word_files = os.listdir(SIMPLE_PATH)
    word_files = [f for f in word_files if "all_unigram_top_1000_100.txt" not in f]
    for word_file in word_files:
        with open(os.path.join(SIMPLE_PATH, word_file), "r") as promptf:
            all_words = promptf.readlines()

            for _ in range(len(all_words)):
                texts_A = np.random.choice(all_words, text_len)
                while len(texts_A) != len(set(texts_A)):
                    texts_A = np.random.choice(all_words, text_len)
                texts_B = []
                for _ in range(text_len):
                    text_B = np.random.choice(all_words)
                    while text_B in texts_A or text_B in texts_B:
                        text_B = np.random.choice(all_words)
                    texts_B.append(text_B)
                text_A = (" ".join([t_A.strip() for t_A in texts_A])).strip()
                text_B = (" ".join([t_B.strip() for t_B in texts_B])).strip()

                template = np.random.choice(prompt_templates).strip()
                for _ in range(n_samples_per_prompt):
                    prompts_A.append(
                        {
                            "text": text_A.strip(),
                            "prompt": template.replace('""', f'"{text_A}"'),
                        }
                    )
                    prompts_B.append(
                        {
                            "text": text_B.strip(),
                            "prompt": template.replace('""', f'"{text_B}"'),
                        }
                    )
                    prompts_AB.append(
                        {
                            "text": text_B.strip(),
                            "prompt": template.replace('""', f'"{text_B}"'),
                        }
                    )
                    templates_A.append(template)
                    templates_B.append(template)

    assert len(prompts_A) == len(prompts_B)
    assert len(prompts_A) == 300 * n_samples_per_prompt
    assert len(prompts_B) == 300 * n_samples_per_prompt
    return prompts_A, prompts_B, prompts_AB, templates_A, templates_B
