# https://pysource.com/blur-faces-in-real-time-with-opencv-mediapipe-and-python
import mediapipe as mp
import cv2
import numpy as np


class FaceLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()

    def get_facial_landmarks(self, cv2_image):
        height, width, _ = cv2_image.shape
        cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(cv2_image_rgb)

        facelandmarks = []
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])
        return np.array(facelandmarks, np.int32)
