import copy

def get_p_r_acc(pred, gt):

    pred = [p.strip().lower() for p in pred]
    gt = [g.strip().lower() for g in gt]

    pred_orig = copy.deepcopy(pred)
    gt_orig = copy.deepcopy(gt)

    pred_length = len(pred)
    gt_length = len(gt)

    for p in pred:
        if p in gt_orig:
            pred_orig.remove(p)
            gt_orig.remove(p)

    p = (pred_length - len(pred_orig)) / (pred_length + 1e-8)
    r = (gt_length - len(gt_orig)) / (gt_length + 1e-8)

    pred_sorted = sorted(pred)
    gt_sorted = sorted(gt)
    if "".join(pred_sorted) == "".join(gt_sorted):
        acc = 1
    else:
        acc = 0

    return p, r, acc


def get_ocr_easyocr(use_cuda=False):
    import easyocr

    return easyocr.Reader(["en"], gpu=use_cuda)


def get_text_easyocr(ocr_model, img):
    texts = ocr_model.readtext(img, detail=0)
    return " ".join(texts)


def ocr_metrics(pred_texts, gt_texts):
    assert len(pred_texts) == len(gt_texts)

    pr, rec, acc = [], [], []
    for pt, gt in zip(pred_texts, gt_texts):
        p, r, a = get_p_r_acc(pt, gt)
        pr.append(p)
        rec.append(r)
        acc.append(a)
    return pr, rec, acc
