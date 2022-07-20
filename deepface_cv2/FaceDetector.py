import os
import math

import dlib
import cv2
import numpy as np
from PIL import Image

DLIB_FACE_DETECTOR = dlib.get_frontal_face_detector()

CV2_FACE_DETECOR = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,
                                                      "haarcascade_frontalface_default.xml"))

CV2_EYE_DETECTOR = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,
                                                      "haarcascade_eye.xml"))


def cv2_align_face(img):
    # eye detector expects gray scale image
    detected_face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # eyes = eye_detector.detectMultiScale(detected_face_gray, 1.3, 5)
    eyes = CV2_EYE_DETECTOR.detectMultiScale(detected_face_gray, 1.1, 10)
    # opencv eye detectin module is not strong. it might find more than 2 eyes!
    # besides, it returns eyes with different order in each call (issue 435)
    # this is an important issue because opencv is the default detector and ssd also uses this
    # find the largest 2 eye. Thanks to @thelostpeace
    eyes = sorted(eyes, key=lambda v: abs((v[0] - v[2]) * (v[1] - v[3])), reverse=True)

    if len(eyes) >= 2:
        # decide left and right eye
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2;
            right_eye = eye_1
        # find center of eyes
        left_eye = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        right_eye = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
        img = alignment_procedure(img, left_eye, right_eye)
    return img  # return img anyway


def alignment_procedure(img, left_eye, right_eye):
    # this function aligns given face in img based on left and right eye coordinates
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    # find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        # rotate same direction to clock
        direction = -1
    else:
        point_3rd = (left_eye_x, right_eye_y)
        # rotate inverse direction of clock
        direction = 1
    # find length of triangle edges

    def euclidean_distance(source_representation, test_representation):
        dist = source_representation - test_representation
        dist = np.sum(np.multiply(dist, dist))
        dist = np.sqrt(dist)
        return dist
    a = euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = euclidean_distance(np.array(right_eye), np.array(left_eye))

    # apply cosine rule
    # this multiplication causes division by zero in cos_a calculation
    if b != 0 and c != 0:
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        # angle in radian
        angle = np.arccos(cos_a)
        # radian to degree
        angle = (angle * 180) / math.pi
        # rotate base image
        if direction == -1:
            angle = 90 - angle
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    # return img anyway
    return img


