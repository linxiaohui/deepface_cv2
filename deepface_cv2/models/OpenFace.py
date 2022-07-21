# -*- coding: utf-8 -*-
import os

import onnxruntime

from .. import preprocess

OPENFACE_MODEL = onnxruntime.InferenceSession(os.path.join(preprocess.get_deepface_home(),
                                                           "open_face.onnx"))

input_shape = (96, 96)


def predict(img):
    out = OPENFACE_MODEL.run(['norm_layer'], input_feed={'input_1': img})
    return out
