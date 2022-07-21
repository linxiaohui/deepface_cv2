# -*- coding: utf-8 -*-
import os

import cv2

from .. import preprocess

ARCFACE_MODEL = cv2.dnn.readNetFromONNX(os.path.join(preprocess.get_deepface_home(),
                                                     "arcface.onnx"))

input_shape = (112, 112)


def predict(img):
    ARCFACE_MODEL.setInput(img)
    out = ARCFACE_MODEL.forward()
    return out
