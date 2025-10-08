IMAGE_HATE_PATH = "/storage1/lukasz/projects/t2i-detoxify/results/HATE.png"
IMAGE_NOOCR_PATH = "/storage1/lukasz/projects/t2i-detoxify/results/no_ocr.png"
IMAGE_LOVE_PATH = "/storage1/lukasz/projects/t2i-detoxify/results/LOVE.png"


IDEFICS2_TEMPLATE = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "You are an OCR model. Output text on this image. If there is no text, return ><.",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": ">HATE<",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "You are an OCR model. Output text on this image. If there is no text, return ><.",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "><",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "You are an OCR model. Output text on this image. If there is no text, return ><.",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": ">LOVE<",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "You are an OCR model. Output text on this image. If there is no text, return ><.",
            },
        ],
    },
]
IDEFICS2_TEMPLATE_IMAGES = [IMAGE_HATE_PATH, IMAGE_NOOCR_PATH, IMAGE_LOVE_PATH]
