import os
import io
import base64
from pathlib import Path

from PIL import Image
import requests
import numpy as np
import cv2

from . import FaceDetector


def initialize_folder():
    home = get_deepface_home()

    if not os.path.exists(home):
        os.makedirs(home)
        print("Directory ", home, "created")


def get_deepface_home():
    return os.getenv('DEEPFACE_HOME', default=os.path.join(str(Path.home()), ".deepface_cv2"))


def load_base64_image(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def load_image(img):
    """支持np.array、字节流、base64编码的图片、URL、本地文件路径"""
    if type(img).__module__ == np.__name__:
        return img
    elif isinstance(img, bytes):
        nparr = np.fromstring(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    elif len(img) > 11 and img[0:11] == "data:image/":
        img = load_base64_image(img)
        return img
    elif len(img) > 4 and img.startswith("http"):
        img = np.array(Image.open(requests.get(img, stream=True).raw).convert('RGB'))
        return img
    else:
        if not os.path.isfile(img):
            raise ValueError("Confirm that ", img, " exists")
        img = cv2.imread(img)
        return img


def detect_faces(img, detector_backend='dlib', grayscale=False, enforce_detection=True, align=True):
    img_region = [0, 0, img.shape[0], img.shape[1]]
    # people would like to skip detection and alignment if they already have pre-processed images
    if detector_backend == 'skip':
        return [(img, img_region)]
    if detector_backend == 'dlib':
        faces = []
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = FaceDetector.DLIB_FACE_DETECTOR(img, 1)
        for d in detected:
            # img = cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
            left = d.left()
            right = d.right()
            top = d.top()
            bottom = d.bottom()
            detected_face = img[max(0, top): min(bottom, img.shape[0]), max(0, left): min(right, img.shape[1])]
            img_region = [left, top, right - left, bottom - top]
            faces.append((detected_face, img_region))
        return faces
    if detector_backend == 'opencv':
        faces = []
        detected = FaceDetector.CV2_FACE_DETECOR.detectMultiScale(img, 1.1, 10)
        for x, y, w, h in detected:
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]
            img_region = [x, y, w, h]
            if align:
                detected_face = FaceDetector.cv2_align_face(detected_face)
            faces.append((detected_face, img_region))
        return faces
    if detector_backend == 'mtcnn':
        faces = []
        return faces


def normalize_input(img, normalization='base'):
    # issue 131 declares that some normalization techniques improves the accuracy

    if normalization == 'base':
        return img
    else:
        # @trevorgribble and @davedgd contributed this feature

        # restore input in scale of [0, 255]
        # because it was normalized in scale of  [0, 1] in preprocess_face
        img *= 255
        if normalization == 'raw':
            pass  # return just restored pixels

        elif normalization == 'Facenet':
            mean, std = img.mean(), img.std()
            img = (img - mean) / std

        elif normalization == "Facenet2018":
            # simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
            img /= 127.5
            img -= 1

        elif normalization == 'VGGFace':
            # mean subtraction based on VGGFace1 training data
            img[..., 0] -= 93.5940
            img[..., 1] -= 104.7624
            img[..., 2] -= 129.1863

        elif normalization == 'VGGFace2':
            # mean subtraction based on VGGFace2 training data
            img[..., 0] -= 91.4953
            img[..., 1] -= 103.8827
            img[..., 2] -= 131.0912

        elif normalization == 'ArcFace':
            # Reference study: The faces are cropped and resized to 112×112,
            # and each pixel (ranged between [0, 255]) in RGB images is normalised
            # by subtracting 127.5 then divided by 128.
            img -= 127.5
            img /= 128

    return img


def preprocess_faces(img, target_size=(224, 224), grayscale=False, enforce_detection=False,
                     detector_backend='dlib', return_region=False, align=True):
    """return a list of faces with region"""
    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img = load_image(img)
    base_img = img.copy()
    faces = []
    detected = detect_faces(img=img, detector_backend=detector_backend, grayscale=grayscale,
                            enforce_detection=enforce_detection, align=align)

    for img, region in detected:
        # post-processing
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize image to expected shape
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = target_size[0] / img.shape[0]
            factor_1 = target_size[1] / img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)

            # Then pad the other side to the target size by adding black pixels
            diff_0 = target_size[0] - img.shape[0]
            diff_1 = target_size[1] - img.shape[1]
            if not grayscale:
                # Put the base image in the middle of the padded image
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                             'constant')
            else:
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)),
                             'constant')

        # double check: if target image is not still the same size with target.
        if img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)

        # normalizing the image pixels
        # like tensorflow.keras.image.image_to_array
        img_pixels = np.asarray(img, dtype=np.float32)
        if len(img_pixels.shape) == 2:
            img_pixels = img_pixels.reshape((img_pixels.shape[0], img_pixels.shape[1], 1))
        img_pixels = np.expand_dims(img_pixels, axis=0)
        # normalize input in [0, 1]
        img_pixels /= 255

        if return_region:
            faces.append((img_pixels, region))
        else:
            faces.append(img_pixels)

    return faces
