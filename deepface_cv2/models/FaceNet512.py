# -*- coding: utf-8 -*-
import os

import onnxruntime

from .. import preprocess

FACENET512_MODEL = onnxruntime.InferenceSession(os.path.join(preprocess.get_deepface_home(),
                                                "facenet512.onnx"))

input_shape = (160, 160)

# print([x.name for x in FACENET512_MODEL.get_inputs()])
# print([x.name for x in FACENET512_MODEL.get_outputs()])


def predict(img):
    out = FACENET512_MODEL.run(['Bottleneck_BatchNorm'], input_feed={'input_3': img})
    return out
