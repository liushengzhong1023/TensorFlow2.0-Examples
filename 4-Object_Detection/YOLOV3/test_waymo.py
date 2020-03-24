#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:07:27
#   Description :
#
# ================================================================

import cv2
import os
import math
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo_process.parse_frame import convert_bounding_box, extract_image_and_label_from_frame

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----------------------------------------------Load Regular Image------------------------------------------------------
'''
Output is a numpy array of given size: [1, input_size, input_size, 3], batch size can be changed.
'''
# Load tf record
input_size = 416
image_path = "./docs/kite.jpg"

# load data
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

# resize into [input_size, input_size]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

# -----------------------------------------------Load Waymo data--------------------------------------------------------
'''
Output is a numpy array of given size: [1, input_size, input_size, 3], batch size can be changed.
'''
input_file = "/home/sl29/data/Waymo/validation/segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord"

# extract the whole segment, should have 200 frames
video_segment = tf.data.TFRecordDataset(input_file, compression_type='')
frame_list = []
for data in video_segment:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    frame_list.append(frame)
    break
print("Frame count: " + str(len(frame_list)))

for frame in frame_list:
    frame_output = extract_image_and_label_from_frame(frame)
print(frame_output)

# # ---------------------------------------Initialize input layer and YOLOv3 model----------------------------------------
# # NOTE: the shape param does not include the batch size
# input_layer = tf.keras.layers.Input([input_size, input_size, 3])
# feature_maps = YOLOv3(input_layer)
#
# # decode bounding boxes
# bbox_tensors = []
# for i, fm in enumerate(feature_maps):
#     bbox_tensor = decode(fm, i)
#     bbox_tensors.append(bbox_tensor)
#
# model = tf.keras.Model(inputs=input_layer, outputs=bbox_tensors)
# utils.load_weights(model, "./yolov3.weights")
#
# # Print a useful summary of the model
# # model.summary()
#
# # ------------------------------------------------------Perform prediction----------------------------------------------
# pred_bbox = model.predict(image_data)
# pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
# pred_bbox = tf.concat(pred_bbox, axis=0)
# bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
# bboxes = utils.nms(bboxes, 0.45, method='nms')
#
# image = utils.draw_bbox(original_image, bboxes)
# image = Image.fromarray(image)
# image.show()
