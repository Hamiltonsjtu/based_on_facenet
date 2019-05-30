import cv2
import numpy as np

def visual_img_BoundaryBox(image, boxes):
    """
    image: data
    boxes: normalized boundary box
    """
    im_height, im_width = np.shape(image)[0:2]
    for i, box in enumerate(boxes):
        ymin = box[0]
        xmin = box[1]
        ymax = box[2]
        xmax = box[3]
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = xmin * im_width   # left
        bb[1] = xmax * im_width   # right
        bb[2] = ymin * im_height  # bottom
        bb[3] = ymax * im_height  # top
        img_crop = cv2.rectangle(image, (bb[0], bb[2]), (bb[1], bb[3]), (0, 255, 0))
        cv2.imshow('img_crp', img_crop)
    cv2.waitKey()