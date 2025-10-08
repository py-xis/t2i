import numpy as np
import cv2


# download `DB_IC15_resnet50.onnx` from https://drive.google.com/drive/folders/1qzNCHfUJOS0NEUOIKn69eCtxdlNPpWbq

# recommended parameter setting: -inputHeight=736, -inputWidth=1280;


def setup_text_detection_model(model_path, input_size=(1280, 736)):
    model = cv2.dnn.TextDetectionModel_DB(model_path)

    # Post-processing parameters
    bin_thresh = 0.3
    poly_thresh = 0.5
    max_candidates = 200
    unclip_ratio = 2.0

    model.setBinaryThreshold(bin_thresh)
    model.setPolygonThreshold(poly_thresh)
    model.setMaxCandidates(max_candidates)
    model.setUnclipRatio(unclip_ratio)

    # Normalization parameters
    scale = 1.0 / 255.0
    mean = (122.67891434, 116.66876762, 104.00698793)

    model.setInputParams(scale, input_size, mean)
    return model


def remove_text_boxes(images, model, input_size=(1280, 736)):
    images = images.copy()
    _, output_w, output_h, _ = images.shape
    for i in range(images.shape[0]):
        image = images[i].astype(np.uint8)
        image = cv2.resize(image, input_size)
        det_results = model.detect(image)
        for points in det_results[0]:
            points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [points], color=(0, 0, 0))
        images[i] = cv2.resize(image, (output_w, output_h))
    return images
