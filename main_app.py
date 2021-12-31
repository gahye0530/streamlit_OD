import streamlit as st
import os
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def save_uploaded_file(directory, file) :
    # 1.디렉토리가 있는지 확인하여, 없으면 디렉토리부터만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)
    # 2. 디렉토리가 있으니, 파일을 저장.
    with open(os.path.join(directory, file.name), 'wb') as f :
        f.write(file.getbuffer())
    return st.success("Saved file : {} in {}".format(file.name, directory))

def main() :
    st.sidebar.info('Object Detection')
    dir = 'data'
    upload_img = st.file_uploader('Choose a image(.jpg) file', type = ['jpg'])
    img_path = os.path.join(dir, upload_img.name)
    if upload_img is not None : 
        save_uploaded_file(dir,upload_img)
        # if문 안에 써주지 않으면 에러난다. 
        # save_uploaded_file 에서 file.name 의 name을 찾을 수 없다고 
    print(img_path)
    image_np = load_image_into_numpy_array(img_path)
    print(image_np)

if __name__ == '__main__' : main()

def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

def load_model(model_dir) :
    model_full_dir = model_dir + "/saved_model"  
    # Load saved model and build the detection function
    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

def show_inference(detection_model, image_np) :
    pass
    
def load_image_into_numpy_array(path):
    # return np.array(Image.open(path))
    print(str(path))
    return cv2.imread(str(path))