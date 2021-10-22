
import numpy as np
#import matplotlib.pyplot as plt
import cv2 as cv2
import os
from pathlib import Path
from PIL import Image, ImageFilter

from os import P_DETACH


try:
    import streamlit as st
    import os
    import sys
    import pandas as pd
    from io import BytesIO, StringIO
    print("All module loaded")
except Exception as e:
    print( "Some Modules are Missing : {} ".format(e))

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""




file = st.file_uploader("Upload file", type=["png", "jpg"])
arch = '_blur_data/weights/deploy.prototxt.txt'
weights = '_blur_data/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel'
neural_net = cv2.dnn.readNetFromCaffe(arch, weights)
threshold = 0.4


def main():
    st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload file", type=["png", "jpg"])
    show_file = st.empty()
    
    if not file:
        show_file.info("Please Upload a file : {} ".format('  '.join(["png", "jpg"])))
        return
    content = file.getvalue()

    if isinstance(file, BytesIO):
        #mosaic()
        show_file.image(file)
    else:
        df = pd.read_csv(file)
        st.dataframe(df.head(2))
        file.close()

def load_model():
    arch = 'blur_data/weights/deploy.prototxt.txt'
    weights = 'blur_data/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    net = cv2.dnn.readNetFromCaffe(arch, weights)
    return net


def mosaic():
    image = cv2.imread(file)
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
    image, 
    scalefactor=1.0, 
    size=(300, 300), # Specify the spatial size of the image.
    mean=(103.93, 116.77, 123.68) # Normalize by subtracting the per-channel means of ImageNet images (which were used to train the pre-trained model).
    )

    neural_net.setInput(blob)
    #detections = neural_net.forward()
    output = np.squeeze(neural_net.forward())

    detections = output

    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
    
        if confidence > 0.3:
            box = output[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(np.int)
            
            face = image[start_y: end_y, start_x: end_x]
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            
            image[start_y: end_y, start_x: end_x] = face


main()


