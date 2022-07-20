import time
start_time = time.time()

from deepface import DeepFace
from .. import DeepFace as DeepFaceCV2

print("all loaded", time.time()-start_time)
start_time = time.time()

img_path = "deepface_cv2/tests/dataset/img10.jpg"
print(DeepFace.analyze(img_path, actions=('age', 'emotion', 'gender', 'race')))

print("DeepFace OK", time.time()-start_time)
start_time = time.time()

print(DeepFaceCV2.analyze(img_path, actions=('age', 'emotion', 'gender', 'race')))
print("DeepFace CV2 OK", time.time()-start_time)
start_time = time.time()
