import datetime

import cv2
import numpy as np
from PIL import Image
from torch_mtcnn import detect_faces


def detecting_face_from_pillow_image(image, size_scale=2):
    # standardize detector input from cv2 image to PIL Image
    image = cv2.resize(
        image,
        (image.shape[1] // size_scale, image.shape[0] // size_scale)
    )
    image = Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    # for some reason, detector randomly throws an error
    try:
        bounding_boxes = np.array(detect_faces(image))
    except:
        bounding_boxes = np.array([])
    # postprocess bounding bboxes
    if bounding_boxes.size > 0:
        scores = bounding_boxes[:, -1]
        bounding_boxes = bounding_boxes[:, :-1].astype("int")
    else:
        scores = np.array([])
    return bounding_boxes * size_scale, scores


def locating_centroids(bounding_boxes):
    if bounding_boxes.size == 0:
        return []
    return np.c_[
        (bounding_boxes[:, 0] + bounding_boxes[:, 2]) / 2,
        (bounding_boxes[:, 1] + bounding_boxes[:, 3]) / 2,
    ].astype("int")




def draw_bounding_boxes(frame, bounding_boxes, scores):
    for score, (x1, y1, x2, y2) in zip(scores, bounding_boxes):
        text = f"({score:0.2%})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        #name_file = f"E:/vtcode_doc/CTtracking_object/images/{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
        #cv2.imwrite(name_file,frame[y1:y2,x1:x2])
    return frame
