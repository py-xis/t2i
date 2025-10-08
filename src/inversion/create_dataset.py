import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

N_TO_CHOOSE = 400
PATH_OUT = "real_images_400/"
WIKI_1000_PATH = "/net/storage/pr3/plgrid/plggdiffusion/ds_ls/wiki1000.txt"
LAION_OCR_DIR = "/net/storage/pr3/plgrid/plggdiffusion/datasets/laion_ocr"
WORKING_DIR = "/net/storage/pr3/plgrid/plggdiffusion/ds_ls"

np.random.seed(42)


def has_max1_words(record):
    return len(record["words"].split(" ")) <= 1


def has_text(record):
    return len(record["words"].strip()) > 0


os.chdir(LAION_OCR_DIR)
path_metadata_train = Path(LAION_OCR_DIR) / Path("train/metadata.jsonl")
all_wiki_words = [w.strip() for w in open(WIKI_1000_PATH).readlines()]

lines = []
with open(path_metadata_train, "r") as file:
    for line in file:
        lines.append(json.loads(line))

lines_max1 = list(filter(has_max1_words, lines))
lines_max1 = list(filter(has_text, lines_max1))
lines_max1_wiki = [l for l in lines_max1 if l["words"].strip().lower() in all_wiki_words]
print(f"Number of images with wiki one word: {len(lines_max1_wiki)}")

all_selected = np.random.choice(lines_max1_wiki, size=N_TO_CHOOSE, replace=False).tolist()

path_out = (Path(WORKING_DIR) / Path(PATH_OUT)).resolve()
os.makedirs(path_out, exist_ok=True)

for selected_file in all_selected:
    img_path = (Path(LAION_OCR_DIR) / Path(f"train/{selected_file['file_name']}")).resolve()
    shutil.copy(img_path, path_out)

path_out_metadata = (path_out / "metadata.jsonl").resolve()

all_wiki_words = [w.strip().lower() for w in open(WIKI_1000_PATH).readlines()]
wiki_words_set = np.random.choice(all_wiki_words, size=N_TO_CHOOSE, replace=False).tolist()

with open(path_out_metadata, "w") as file:
    for selected_file in all_selected:
        out_file = {
            "file_name": selected_file["file_name"],
            "prompt": selected_file["text"].strip(),
            "text": selected_file["words"].strip(),
        }
        file.write(json.dumps(out_file) + "\n")

path_out_metadata_edit = (path_out / "metadata_edit.jsonl").resolve()

with open(path_out_metadata_edit, "w") as file:
    for selected_file, wiki_word in zip(all_selected, wiki_words_set):
        prompt_edit = re.sub(r'"[^"]+"', f'"{wiki_word}"', selected_file["text"].strip())
        out_file = {
            "file_name": selected_file["file_name"],
            "prompt": selected_file["text"].strip(),
            "text": selected_file["words"].strip(),
            "prompt_edit": prompt_edit,
            "text_edit": wiki_word,
        }
        file.write(json.dumps(out_file) + "\n")
