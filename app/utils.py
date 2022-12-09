import os
import cv2 
import numpy as np
import pytesseract
import tensorflow as tf
from pytesseract import Output

# ID card classes(front and back)
labels = [{'name':'placeofbirth', 'id':1}, {'name':'dateofbirth', 'id':2},{'name':'height', 'id':3},{'name':'bloodgroup', 'id':4},{'name':'sex', 'id':5},\
          {'name':'expirelocation', 'id':6},{'name':'id1', 'id':7},{'name':'id2', 'id':8},{'name':'idnumber', 'id':9},{'name':'lastnames', 'id':10}, \
          {'name':'firstnames', 'id':11}]

def load_model():
    model_path = os.path.join(os.path.dirname(__file__),'model')
    detect_fn = tf.saved_model.load(model_path)

    return detect_fn

def to_tensor(image):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    return (input_tensor,image_np)

def perform_ocr(detections, image_size, image):
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
        
    detection_threshold = 0.1

    scores = list(filter(lambda x: x >detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    height, width = image_size
    left = int(0.01 * width)  # shape[1] = cols
    right = left
    bottom = top = int(0.01 * height) # shape[0] = rows
    borderType = cv2.BORDER_CONSTANT
    value = [0,0,0]

    ocr_result = []

    for idx, box in enumerate(boxes):
    
        roi = box*[height, width, height, width]

        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        region = cv2.resize( region, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)

            
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        region = cv2.GaussianBlur(region, (5,5), 0)
        region = cv2.medianBlur(region, 3)

        ret, region = cv2.threshold(region, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        region = cv2.copyMakeBorder(region, top, bottom, left, right, borderType, None, value)

        # dilation
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        region = cv2.dilate(region, rect_kern, iterations = 1)

        custom_config = r'-c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+.- " --psm 8'
        result = pytesseract.image_to_data(region,output_type=Output.DICT,config=custom_config)

        ocr_result.append(result)

    return ocr_result, classes


def result_to_dict(ocr_result, classes):
    detected_roi_names = []
    result = {}
    for class_idx in classes:
        for idx, item in enumerate(labels):
            if labels[idx]['id'] == class_idx:
                detected_roi_names.append(labels[idx]['name'])

    for idx, text in enumerate(ocr_result):
            string_value = ""
            for text in text['text']:
                if text:
                    string_value = string_value + text + " "
            string_value = string_value.rstrip('-. ').lstrip('- ')
            result[detected_roi_names[idx]] = string_value

    return result    