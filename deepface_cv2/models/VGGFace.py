# -*- coding: utf-8 -*-
import os

import cv2

from .. import preprocess

VGG_MODEL = cv2.dnn.readNetFromONNX(os.path.join(preprocess.get_deepface_home(),
                                                 "vgg_face.onnx"))

input_shape = (224, 224)


def predict(img):
    VGG_MODEL.setInput(img)
    out = VGG_MODEL.forward()
    return out
