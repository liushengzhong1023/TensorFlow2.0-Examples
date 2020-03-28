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
import time
import argparse
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo_process.parse_frame import extract_image_and_label_from_frame, extract_frame_list
from waymo_process.schedule_frame import serialize_full_frames, serialize_partial_frames, batched_partial_frames, \
    prioritize_serialize_partial_frames
from waymo_process.partial_frame_postprocess import postprocess_box_one_batch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument("-scheduling_policy", type=str,
                    default="prioritize_serialize_partial_frames",
                    help="The choice of scheduling policy.")
args = parser.parse_args()

# -----------------------------------------------Load warm up image-----------------------------------------------------
warmup_input_size = 416
warmup_image_path = "./docs/kite.jpg"

original_image = cv2.imread(warmup_image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

warmup_image_data, _, _ = utils.image_preprocess(np.copy(original_image), [warmup_input_size, warmup_input_size])
warmup_image_data = warmup_image_data[np.newaxis, ...].astype(np.float32)

# -----------------------------------------------Load Waymo data--------------------------------------------------------
'''
Output is a numpy array of given size: [1, input_size, input_size, 3], batch size can be changed.
'''
input_file = "/home/sl29/data/Waymo/validation/segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord"

# Extract the whole segment, should have 200 frames
start = time.time()
frame_list = extract_frame_list(input_file, use_single_camera=True)
frame_count = len(frame_list)
end = time.time()
print("------------------------------------------------------------------------")
print("Frame count: " + str(frame_count))
print("File reading and parsing time: %f s" % (end - start))

# ---------------------------------------Initialize input layer and YOLOv3 model----------------------------------------
# NOTE: the shape param does not include the batch size
input_layer = tf.keras.layers.Input([None, None, 3])
feature_maps = YOLOv3(input_layer)

# decode bounding boxes
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(inputs=input_layer, outputs=bbox_tensors)
utils.load_weights(model, "./yolov3.weights")

# # ---------------------------------------------Scheduling & Inference-------------------------------------------------
'''
Each frame is scheduled and predicted in serialized order; the scheduling within each frame is considered.

'''
# warm up run
for _ in range(5):
    pred_bbox = model.predict(warmup_image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, [warmup_input_size, warmup_input_size], 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

start = time.time()

for extracted_frame in frame_list:
    if args.scheduling_policy == "serialize_full_frames":
        image_queue = serialize_full_frames(extracted_frame)
    elif args.scheduling_policy == "serialize_partial_frames":
        image_queue = serialize_partial_frames(extracted_frame)
    elif args.scheduling_policy == "prioritize_serialize_partial_frames":
        image_queue, meta_queue = prioritize_serialize_partial_frames(extracted_frame)
    else:
        image_queue = batched_partial_frames(extracted_frame)
    end = time.time()
    print("------------------------------------------------------------------------")
    print('Batch count: ' + str(len(image_queue)))
    # print("Scheduling time: %f s" % (end - start))

    for (i, image_batch) in enumerate(image_queue):
        pred_bbox = model.predict(image_batch)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        if args.scheduling_policy == "prioritize_serialize_partial_frames":
            frame_meta = meta_queue[i]
            processed_bboxes = postprocess_box_one_batch(pred_bbox, frame_meta)

end = time.time()
print("------------------------------------------------------------------------")
print("Total inference time: %f s" % (end - start))
print("Inference time per frame: %f s" % ((end - start) / frame_count))
