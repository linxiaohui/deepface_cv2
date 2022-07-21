# -*- coding: utf-8 -*-
import os

import cv2

from .. import preprocess

DEEPID_MODEL = cv2.dnn.readNetFromONNX(os.path.join(preprocess.get_deepface_home(),
                                                    "deepid.onnx"))
# input_shape = (55, 47)
input_shape = (47, 55)


def predict(img):
    DEEPID_MODEL.setInput(img)
    out = DEEPID_MODEL.forward()
    return out
