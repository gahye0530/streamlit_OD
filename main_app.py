import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# from object_detection.utils import ops as utils_ops
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def load_image_into_numpy_array(path):
    # return np.array(Image.open(path))
    print(str(path))
    return cv2.imread(str(path))

def load_model(model_dir) :
    model_full_dir = model_dir + "/saved_model"  
    # Load saved model and build the detection function
    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

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
    
    # SSD MobileNet v2 320*320 : /20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
    MODEL_DATE = '20200711'
    MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
    PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

    # 카테고리인덱스 경로 설정
    PATH_TO_LABELS = 'C:\\Users\\gahye\\OneDrive\\Documents\\Tensorflow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)
    # 모델 로드
    detection_model = load_model(PATH_TO_MODEL_DIR)
    # 이미지 넘파이 어레이로 변경
    image_np = load_image_into_numpy_array(img_path)

    ## 학습
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detection_model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    # show 
    cv2.imshow(str(img_path), image_np_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__' : main()

def show_inference(detection_model, image_np) :
    pass