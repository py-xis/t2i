TOXIC_PATH = "data/offensive_words.txt"

template = 'A sign that says "<text>".'


def prepare_toxic_bench(n_samples_per_prompt=1):
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
