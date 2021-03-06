##ADD_CIRCLE BLURRING

import PIL
import streamlit as st
import numpy as np
import cv2
import os
from os.path import dirname, join
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import math

from os import P_DETACH


arch = join(dirname(__file__), "blur_data/weights/deploy.prototxt.txt")
weights = join(dirname(__file__), "blur_data/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel")
#arch = 'blur_data/weights/deploy.prototxt.txt'
#weights = 'blur_data/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel'
neural_net = cv2.dnn.readNetFromCaffe(arch, weights)

threshold = 0.3

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

            #test
            p1 = (65, 65)
            w, h = 100, 100
            p2 = (p1[0] + w, p1[1] + h)


            circle_center = ((p1[0] + p2[0])// 2, (p1[1] + p2[1]) // 2)
            circle_radius = int(math.sqrt(w * w + h * h) // 2)
            mask_img = np.zeros(img.shape, dtype='uint8')
            cv2.circle(mask_img, circle_center, circle_radius, (255, 255, 255), -1)

            img_all_blurred = cv2.medianBlur(img, 99)
            img_face_blurred = np.where(mask_img > 0, img_all_blurred, img)
            #test

        
            face = cv2_image[start_y: end_y, start_x: end_x]
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
        
            cv2_image[start_y: end_y, start_x: end_x] = face
    
    cv2_image = Image.fromarray(cv2_image)



    return cv2_image



def main():

    st.title("Face Blurring App")
    st.text("blurring face using openCV")
    st.text("Built via Streamlit")


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
        task = ['Original','Blurring']
        feature_choice = st.sidebar.selectbox("Function", task)
        if st.button("Process"):

            if feature_choice == 'Original':
                pass
            elif feature_choice == 'Blurring':
                #st.write(type(cv2_image))
                #st.image(detect_faces(cv2_image))
                #cv2_image = np.array(cv2_image)
                st.image(detect_faces(cv2_image))
                

        


    elif choice == 'About':
        st.subheader('About')


if __name__ == '__main__':
    main()

