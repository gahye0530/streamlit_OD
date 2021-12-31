import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from obj_detection_func import run_obj_detection
# from object_detection.utils import ops as utils_ops
import tensorflow as tf
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

def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

def main() :
    st.sidebar.info('Object Detection')
    st.subheader('Object Detector Select')
    model_dic = {'CenterNet Resnet50 V1 FPN 512x512/speed : 27' : ['20200711', 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'],
                 'CenterNet MobileNetV2 FPN 512x512/speed : 6' : ['20210210', 'centernet_mobilenetv2fpn_512x512_coco17_od'],
                 'EfficientDet D0 512x512/speed : 39' : ['20200711', 'efficientdet_d0_coco17_tpu-32'],
                 'SSD MobileNet v2 320x320/speed : 19' : ['20200711', 'ssd_mobilenet_v2_320x320_coco17_tpu-8']}
    model_selected = st.selectbox('', model_dic.keys())
    MODEL_DATE = model_dic[model_selected][0]
    MODEL_NAME = model_dic[model_selected][1]
    st.header('')
    st.header('')
    st.subheader('File Upload(.jpg)')
    dir = 'data'
    upload_img = st.file_uploader('Choose a image file', type = ['jpg'])
    if upload_img is not None : 
        img_path = os.path.join(dir, upload_img.name)
        save_uploaded_file(dir,upload_img)

        # 모델 다운로드
        PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)
        if st.button('Object Detection Execute') : run_obj_detection(PATH_TO_MODEL_DIR, img_path)
if __name__ == '__main__' : main()