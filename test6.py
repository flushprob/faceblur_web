##Last TEST 21-10-21 - 03:43


import PIL
import streamlit as st
import numpy as np
import cv2
import os
from os.path import dirname, join
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from facial_landmarks import FaceLandmarks
import mediapipe as mp

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=True, max_num_faces=40)
mpDraw = mp.solutions.drawing_utils

# from os import P_DETACH
fl = FaceLandmarks()

arch = join(dirname(__file__), "blur_data/weights/deploy.prototxt.txt")
weights = join(dirname(__file__), "blur_data/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel")
# arch = 'blur_data/weights/deploy.prototxt.txt'
# weights = 'blur_data/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel'
neural_net = cv2.dnn.readNetFromCaffe(arch, weights)

threshold = 0.3


def lmsblur(img):
    img = np.array(img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)

    img = Image.fromarray(img)

    return img


def lmsblur2(img):
    img = np.array(img)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img_copy = img.copy()
    height, width, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:

            convexhull = cv2.convexHull(results)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, convexhull, 255)
            # img_copy = cv2.blur(img_copy, (27, 27))
            img_copy = cv2.Blur(img_copy, (27, 27), 0)
            face_extracted = cv2.bitwise_and(img_copy, img_copy, mask=mask)

            background_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(img, img, mask=background_mask)

            result = cv2.add(background, face_extracted)

    img = Image.fromarray(result)

    return img

def lmsblur3(img):
    img = np.array(img)
    landmarks = fl.get_facial_landmarks(img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    height, width, _ = img.shape
    h, w = img.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            convexhull = cv2.convexHull(landmarks)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, convexhull, 255)
            img_copy = cv2.GaussianBlur(img, (kernel_width, kernel_height), 0)
            face_extracted = cv2.bitwise_and(img_copy, img_copy, mask=mask)
            background_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(img, img, mask=background_mask)

            cv2_image = cv2.add(background, face_extracted)
    return cv2_image


def detect_faces(img):
    cv2_image = np.array(img)
    h, w = cv2_image.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    blob = cv2.dnn.blobFromImage(
        cv2_image,
        scalefactor=1.0,
        size=(300, 300),  # Specify the spatial size of the image.
        mean=(103.93, 116.77, 123.68)
        # Normalize by subtracting the per-channel means of ImageNet images (which were used to train the pre-trained model).
    )
    neural_net.setInput(blob)
    # detections = neural_net.forward()
    output = np.squeeze(neural_net.forward())

    for i in range(0, output.shape[0]):
        confidence = output[i, 2]

        if confidence > 0.5:
            box = output[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(np.int)

            face = cv2_image[start_y: end_y, start_x: end_x]
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)

            cv2_image[start_y: end_y, start_x: end_x] = face

    cv2_image = Image.fromarray(cv2_image)

    return cv2_image


def main():
    st.title("Face Blurring App")
    st.text("blurring face using openCV")

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Function", activities)

    if choice == 'Detection':
        st.subheader('Face Blurring')

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            cv2_image = Image.open(image_file)
            st.text("Original Image")
            st.write(type(cv2_image))
            st.image(cv2_image)

        # Face Detection
        task = ['Original', 'Blurring', 'Landmark Blurring - ONE PERSON']
        feature_choice = st.sidebar.selectbox("Function", task)
        if st.button("Process"):

            if feature_choice == 'Original':
                pass
            elif feature_choice == 'Blurring':
                # st.write(type(cv2_image))
                # st.image(detect_faces(cv2_image))
                # cv2_image = np.array(cv2_image)

                st.image(detect_faces(cv2_image))
            elif feature_choice == 'Landmark Blurring - ONE PERSON':
                st.image(lmsblur3(cv2_image))





    elif choice == 'About':
        st.subheader('About')


if __name__ == '__main__':
    main()
