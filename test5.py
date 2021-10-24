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

#from os import P_DETACH
fl = FaceLandmarks()

arch = join(dirname(__file__), "blur_data/weights/deploy.prototxt.txt")
weights = join(dirname(__file__), "blur_data/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel")
#arch = 'blur_data/weights/deploy.prototxt.txt'
#weights = 'blur_data/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel'
neural_net = cv2.dnn.readNetFromCaffe(arch, weights)

threshold = 0.3
def lmsblur(img):
    cv2_image = np.array(img)
    cv2_image = cv2_image
    # cv2_image = cv2.resize(cv2_image, None, fx=0.5, fy=0.5)
    cv2_image_copy = cv2_image.copy()
    height, width, _ = cv2_image.shape
    h, w = cv2_image.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1

    # 1. Face landmarks detection
    landmarks = fl.get_facial_landmarks(cv2_image)





    convexhull = cv2.convexHull(landmarks)
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [convexhull], True, 255, 3)
    cv2.fillConvexPoly(mask, convexhull, 255)

    # Extract the face
    # if landmarks.multi_face_landmarks:
    #     for i in landmarks.multi_face_landmarks:

    #mask = cv2.GaussianBlur(mask, (kernel_width, kernel_height), 0)
    #cv2_image_copy = cv2.blur(cv2_image_copy, (27, 27))
    cv2_image_copy = cv2.GaussianBlur(cv2_image_copy, (kernel_width, kernel_height), 0)
    #cv2_image_copy = cv2.blur(mask, (27, 27))
    face_extracted = cv2.bitwise_and(cv2_image_copy, cv2_image_copy, mask=mask)



    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(cv2_image, cv2_image, mask=background_mask)

    result = cv2.add(background, face_extracted)


    return result


def detect_faces(img):
    cv2_image = np.array(img)
    h, w = cv2_image.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    blob = cv2.dnn.blobFromImage(
    cv2_image,
    scalefactor=1.0,
    size=(300, 300), # Specify the spatial size of the image.
    mean=(103.93, 116.77, 123.68) # Normalize by subtracting the per-channel means of ImageNet images (which were used to train the pre-trained model).
    )
    neural_net.setInput(blob)
    #detections = neural_net.forward()
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
        task = ['Original','Blurring', 'Smooth Blurring - ONLY ONE PERSON']
        feature_choice = st.sidebar.selectbox("Function", task)
        if st.button("Process"):

            if feature_choice == 'Original':
                pass
            elif feature_choice == 'Blurring':
                #st.write(type(cv2_image))
                #st.image(detect_faces(cv2_image))
                #cv2_image = np.array(cv2_image)
                st.image(detect_faces(cv2_image))
            elif feature_choice == 'Smooth Blurring - ONLY ONE PERSON':
                st.image(lmsblur(cv2_image))





    elif choice == 'About':
        st.subheader('About')


if __name__ == '__main__':
    main()

