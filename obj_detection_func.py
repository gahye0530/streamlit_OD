import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
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

def run_obj_detection(PATH_TO_MODEL_DIR, img_path, min_score_slider) :
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
          min_score_thresh=(min_score_slider/100),
          agnostic_mode=False)

    # show 
    # channels를 설정해 주지 않으면 빨간색을 인식하지 못한다 -> 정홍근님 tip
    st.image(image_np_with_detections, channels='BGR')