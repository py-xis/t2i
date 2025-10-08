import os

import numpy as np
from PIL import Image


def extract_text_inside_brackets(s):
    start = s.find(">") + 1
    end = s.find("<", start)
    return s[start:end]


def get_ocr_easyocr(use_cuda=False):
    import easyocr

    return easyocr.Reader(["en"], gpu=use_cuda)


def get_text_easyocr(ocr_model, pil_img):
    img = np.asarray(pil_img)
    texts = ocr_model.readtext(img, detail=0)
    return " ".join(texts)


def get_ocr_surya(use_cuda=False):
    # BAD PERFORMANCE

    from surya.model.detection import model
    from surya.model.recognition.model import load_model
    from surya.model.recognition.processor import load_processor

    os.environ["TORCH_DEVICE"] = "cpu" if not use_cuda else "cuda"

    det_processor, det_model = model.load_processor(), model.load_model()
    rec_model, rec_processor = load_model(), load_processor()
    return det_processor, det_model, rec_model, rec_processor


def get_test_surya(det_processor, det_model, rec_model, rec_processor, pil_img):
    from surya.ocr import run_ocr

    langs = ["en"]
    predictions = run_ocr(
        [pil_img], [langs], det_model, det_processor, rec_model, rec_processor
    )
    return predictions


def get_idefics2(device="cpu"):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        size={"longest_edge": 448, "shortest_edge": 378},
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceM4/idefics2-8b", torch_dtype=torch.float16
    ).to(device)

    return processor, model


def get_text_idefics2(idefics2_processor, idefics2_model, pil_img, device="cpu"):
    from transformers.image_utils import load_image

    from src.eval.constants import IDEFICS2_TEMPLATE, IDEFICS2_TEMPLATE_IMAGES

    prompt_images = IDEFICS2_TEMPLATE_IMAGES + [pil_img]
    prompt_images = [load_image(img) for img in prompt_images]

    prompt = idefics2_processor.apply_chat_template(
        IDEFICS2_TEMPLATE, add_generation_prompt=True
    )
    inputs = idefics2_processor(text=prompt, images=prompt_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = idefics2_model.generate(**inputs, max_new_tokens=100)
    generated_texts = idefics2_processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return extract_text_inside_brackets(generated_texts[0])


if __name__ == "__main__":
    from tqdm import tqdm

    USE_CUDA = True
    TEST_IDEFICS = False
    TEST_EASYOCR = True

    EVAL_DATASET = [
        "/storage1/lukasz/projects/t2i-detoxify/data/dreambooth/dataset/bear_plushie/04.jpg",
        "/storage1/lukasz/projects/t2i-detoxify/data/dreambooth/dataset/backpack_dog/01.jpg",
        "/storage1/lukasz/projects/t2i-detoxify/res/trained-sd3/image_1.png",
        "/storage1/lukasz/projects/t2i-detoxify/results/ex.png",
    ]

    if TEST_IDEFICS:
        print("Testing Idefics2:")
        processor, model = get_idefics2(USE_CUDA)
        for image in tqdm(EVAL_DATASET):
            ret = get_text_idefics2(
                processor,
                model,
                Image.open(image),
                USE_CUDA,
            )
            print(ret)

    if TEST_EASYOCR:
        print("Testing EasyOCR:")
        model = get_ocr_easyocr(USE_CUDA)
        for image in tqdm(EVAL_DATASET):
            ret = get_text_easyocr(model, Image.open(image))
            print(ret)
