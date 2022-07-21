# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

from . import preprocess
from .models import VGGFace, OpenFace, FaceNet, FaceNet512, DeepID, ArcFace

EMOTION_MODEL = cv2.dnn.readNetFromONNX(os.path.join(preprocess.get_deepface_home(),
                                                     "facial_expression_model.onnx"))
AGE_MODEL = cv2.dnn.readNetFromONNX(os.path.join(preprocess.get_deepface_home(),
                                                 "age_model.onnx"))
GENDER_MODEL = cv2.dnn.readNetFromONNX(os.path.join(preprocess.get_deepface_home(),
                                                    "gender_model.onnx"))
RACE_MODEL = cv2.dnn.readNetFromONNX(os.path.join(preprocess.get_deepface_home(),
                                                  "race_model.onnx"))


def analyze(img_path, actions=('emotion', 'age', 'gender', 'race'), models=None,
            enforce_detection=True, detector_backend='opencv', prog_bar=True):
    """
    Analyzes facial attributes including age, gender, emotion and race

    Parameters:
        img_path: image path(local or http), numpy array (BGR), image binary data or base64 encoded image could be passed.
                  **Be noticed**: It can not be a list, which is different from the original one

        actions (tuple): The default is ('age', 'gender', 'emotion', 'race').
                You can drop some of those attributes.

        models: (Optional[dict]) facial attribute analysis models are built in every call of analyze function.
                You can pass pre-built models to speed the function up.
            models = {}
            models['age'] = DeepFace.build_model('Age')
            models['gender'] = DeepFace.build_model('Gender')
            models['emotion'] = DeepFace.build_model('Emotion')
            models['race'] = DeepFace.build_model('Race')

        enforce_detection (boolean): The function throws exception if no face detected by default.
                Set this to False if you don't want to get exception.
                 This might be convenient for low resolution images.

        detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib.

        prog_bar (boolean): enable/disable a progress bar

    Returns:
        The function returns a dictionary. If img_path is a list, then it will return list of dictionary.
        {
            "region": {'x': 230, 'y': 120, 'w': 36, 'h': 45},
            "age": 28.66,
            "dominant_gender": "Woman",
            "gender": {
                'Woman': 99.99407529830933,
                'Man': 0.005928758764639497,
            }
            "dominant_emotion": "neutral",
            "emotion": {
                'sad': 37.65260875225067,
                'angry': 0.15512987738475204,
                'surprise': 0.0022171278033056296,
                'fear': 1.2489334680140018,
                'happy': 4.609785228967667,
                'disgust': 9.698561953541684e-07,
                'neutral': 56.33133053779602
            }
            "dominant_race": "white",
            "race": {
                'indian': 0.5480832420289516,
                'asian': 0.7830780930817127,
                'latino hispanic': 2.0677512511610985,
                'black': 0.06337375962175429,
                'middle eastern': 3.088453598320484,
                'white': 93.44925880432129
            }
        }
    """
    if isinstance(img_path, list):
        raise ValueError("input parameter `img_path` can not be a list")

    gray_faces = []
    faces = []
    if actions and 'emotion' in list(actions):
        gray_faces = preprocess.preprocess_faces(img=img_path, target_size=(48, 48), grayscale=True,
                                                 enforce_detection=enforce_detection,
                                                 detector_backend=detector_backend,
                                                 return_region=True)
    if actions and set(actions) & set(['age', 'gender', 'race']):
        faces = preprocess.preprocess_faces(img=img_path, target_size=(224, 224), grayscale=False,
                                            enforce_detection=enforce_detection,
                                            detector_backend=detector_backend,
                                            return_region=True)
    face_num = max(len(gray_faces), len(faces))
    resp_objects = []
    region_labels = ['x', 'y', 'w', 'h']
    for i in range(face_num):
        resp_obj = {}
        if 'emotion' in list(actions):
            emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            img, region = gray_faces[i]
            resp_obj["region"] = dict(zip(region_labels, region))
            EMOTION_MODEL.setInput(img)
            out = EMOTION_MODEL.forward()
            emotion_predictions = out[0, :]
            resp_obj["emotion"] = dict(zip(emotion_labels, 100*emotion_predictions/emotion_predictions.sum()))
            resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]
        if 'age' in list(actions):
            img, region = faces[i]
            resp_obj["region"] = dict(zip(region_labels, region))
            AGE_MODEL.setInput(img)
            out = AGE_MODEL.forward()
            age_predictions = out[0, :]
            # print(age_predictions)
            output_indexes = np.arange(101)
            apparent_age = np.sum(age_predictions * output_indexes)
            resp_obj["age"] = int(apparent_age)
        if 'gender' in list(actions):
            img, region = faces[i]
            resp_obj["region"] = dict(zip(region_labels, region))
            GENDER_MODEL.setInput(img)
            out = GENDER_MODEL.forward()
            gender_predictions = out[0, :]
            # print(age_predictions)
            if np.argmax(gender_predictions) == 0:
                gender = "Woman"
            elif np.argmax(gender_predictions) == 1:
                gender = "Man"
            resp_obj["gender"] = gender
        if 'race' in list(actions):
            img, region = faces[i]
            resp_obj["region"] = dict(zip(region_labels, region))
            RACE_MODEL.setInput(img)
            out = RACE_MODEL.forward()
            race_predictions = out[0, :]
            race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
            resp_obj["race"] = dict(zip(race_labels, 100*race_predictions / race_predictions.sum()))
            resp_obj["dominant_race"] = emotion_labels[np.argmax(emotion_predictions)]

        resp_objects.append(resp_obj)

    resp_obj = {}
    for i in range(0, len(resp_objects)):
        resp_item = resp_objects[i]
        resp_obj["instance_%d" % (i + 1)] = resp_item
    return resp_obj


def verify(img1_path, img2_path='', model_name='VGG-Face', distance_metric='cosine',
           model=None, enforce_detection=True, detector_backend='opencv', align=True,
           prog_bar=True, normalization='base'):
    """
    This function verifies an image pair is same person or different persons.

    Parameters:
        img1_path, img2_path: exact image path, numpy array (BGR) or based64 encoded images could be passed.
        If you are going to call verify function for a list of image pairs, then you should pass an array instead of calling the function in for loops.

        e.g. img1_path = [
            ['img1.jpg', 'img2.jpg'],
            ['img2.jpg', 'img3.jpg']
        ]

        model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or Ensemble

        distance_metric (string): cosine, euclidean, euclidean_l2

        model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.

            model = DeepFace.build_model('VGG-Face')

        enforce_detection (boolean): If no face could not be detected in an image, then this function will return exception by default. Set this to False not to have this exception. This might be convenient for low resolution images.

        detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

        prog_bar (boolean): enable/disable a progress bar

    Returns:
        Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.

        {
            "verified": True
            , "distance": 0.2563
            , "max_threshold_to_verify": 0.40
            , "model": "VGG-Face"
            , "similarity_metric": "cosine"
        }

    """
    pass


def find(img_path, db_path, model_name='VGG-Face', distance_metric='cosine', model=None, enforce_detection=True,
         detector_backend='opencv', align=True, prog_bar=True, normalization='base', silent=False):
    """
    This function applies verification several times and find an identity in a database

    Parameters:
        img_path: exact image path, numpy array (BGR) or based64 encoded image. If you are going to find several identities, then you should pass img_path as array instead of calling find function in a for loop. e.g. img_path = ["img1.jpg", "img2.jpg"]

        db_path (string): You should store some .jpg files in a folder and pass the exact folder path to this.

        model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble

        distance_metric (string): cosine, euclidean, euclidean_l2

        model: built deepface model. A face recognition models are built in every call of find function. You can pass pre-built models to speed the function up.

            model = DeepFace.build_model('VGG-Face')

        enforce_detection (boolean): The function throws exception if a face could not be detected. Set this to True if you don't want to get exception. This might be convenient for low resolution images.

        detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

        prog_bar (boolean): enable/disable a progress bar

    Returns:
        This function returns pandas data frame. If a list of images is passed to img_path, then it will return list of pandas data frame.
    """


def represent(img_path, model_name='VGG-Face', model=None, enforce_detection=True,
              detector_backend='opencv', align=True, normalization='base'):
    """
    This function represents facial images as vectors.

    Parameters:
        img_path: exact image path, numpy array (BGR) or based64 encoded images could be passed.

        model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace.

        model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times. Consider to pass model if you are going to call represent function in a for loop.

            model = DeepFace.build_model('VGG-Face')

        enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

        detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

        align:

        normalization (string): normalize the input image before feeding to model

    Returns:
        Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
    """
    embeddings = []
    if model is None:
        if model_name == 'VGG-Face':
            model = VGGFace
        elif model_name == 'OpenFace':
            model = OpenFace
        elif model_name == 'Facenet':
            model = FaceNet
        elif model_name == 'Facenet512':
            model = FaceNet512
        elif model_name == 'DeepFace':
            model = OpenFace
        elif model_name == 'DeepID':
            model = DeepID
        elif model_name == 'Dlib':
            model = OpenFace
        elif model_name == 'ArcFace':
            model = ArcFace
        elif model_name == 'SFace':
            model = OpenFace
        else:
            raise Exception("Unsupported Model")
    input_shape_x, input_shape_y = model.input_shape

    # detect and align
    faces = preprocess.preprocess_faces(img=img_path, target_size=(input_shape_y, input_shape_x),
                                        enforce_detection=enforce_detection,
                                        detector_backend=detector_backend,
                                        align=align, return_region=False)
    for img in faces:
        img = preprocess.normalize_input(img=img, normalization=normalization)
        embedding = model.predict(img)[0].tolist()
        embeddings.append(embedding)
    return embeddings

def stream(db_path='', model_name='VGG-Face', detector_backend='opencv', distance_metric='cosine',
           enable_face_analysis=True, source=0, time_threshold=5, frame_threshold=5):
    """
    This function applies real time face recognition and facial attribute analysis

    Parameters:
        db_path (string): facial database path. You should store some .jpg files in this folder.

        model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble

        detector_backend (string): opencv, ssd, mtcnn, dlib, retinaface

        distance_metric (string): cosine, euclidean, euclidean_l2

        enable_facial_analysis (boolean): Set this to False to just run face recognition

        source: Set this to 0 for access web cam. Otherwise, pass exact video path.

        time_threshold (int): how many second analyzed image will be displayed

        frame_threshold (int): how many frames required to focus on face

    """


def detectFace(img_path, target_size=(224, 224), detector_backend='opencv', enforce_detection=True, align=True):
    """
    This function applies pre-processing stages of a face recognition pipeline including detection and alignment

    Parameters:
        img_path: exact image path, numpy array (BGR) or base64 encoded image

        detector_backend (string): face detection backends are retinaface, mtcnn, opencv, ssd or dlib

    Returns:
        deteced and aligned face in numpy format
    """
