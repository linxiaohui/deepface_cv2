# -*- coding: utf-8 -*-
import os

import onnxruntime

from .. import preprocess

FACENET_MODEL = onnxruntime.InferenceSession(os.path.join(preprocess.get_deepface_home(),
                                                          "facenet.onnx"))

input_shape = (160, 160)

# print([x.name for x in FACENET_MODEL.get_inputs()])
# print([x.name for x in FACENET_MODEL.get_outputs()])


def predict(img):
    out = FACENET_MODEL.run(['Bottleneck_BatchNorm'], input_feed={'input_2': img})
    return out
