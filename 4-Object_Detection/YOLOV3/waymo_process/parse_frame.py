import numpy as np
import math
import cv2
import time
import tensorflow as tf

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_process.compute_risk import compute_risk_for_laser_objects, waymo_find_best_match_id

import core.utils as utils

'''
Object class mapping:
      enum Type {
        TYPE_UNKNOWN = 0;
        TYPE_VEHICLE = 1;
        TYPE_PEDESTRIAN = 2;
        TYPE_SIGN = 3;
        TYPE_CYCLIST = 4;
      }
'''


def convert_bounding_box(box):
    '''
    Convert from [center_x, center_y, length, width] --> [x_min, y_min, x_max, y_max]
    '''
    return [box.center_x - 0.5 * box.length, box.center_y - 0.5 * box.width,
            box.center_x + 0.5 * box.length, box.center_y + 0.5 * box.width]


def parse_image_from_buffer(image_buffer):
    '''
    Serialized image buffer --> numpy array
    '''
    # print(image_buffer)
    image = np.fromstring(image_buffer, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def extract_image_and_label_from_frame(frame, use_single_camera=False):
    '''
    Extract the 2d image, and its corresponding labels from the given frame.
    NOTE: 5 cameras deployed (front, front left, front right, side left, side right); JPEG image for each camera.
    output: Dict of 5 dict, where the first level key is image name;
            The second level dict['image']=image, dict['bbox_list']=list of bounding boxes
    '''
    parsed_frame = dict()
    frame_id = frame.timestamp_micros

    # process front camera only
    if use_single_camera:
        target_camera_names = {open_dataset.CameraName.FRONT}
    else:
        target_camera_names = {open_dataset.CameraName.FRONT,
                               open_dataset.CameraName.FRONT_LEFT,
                               open_dataset.CameraName.FRONT_RIGHT,
                               open_dataset.CameraName.SIDE_LEFT,
                               open_dataset.CameraName.SIDE_RIGHT}

    # get risks for objects from laser labels
    laser_object_risks = compute_risk_for_laser_objects(frame)

    # extract the image for each camera
    for image in frame.images:
        image_camera_name = image.name

        # Skip cameras with unknown camera names
        if image_camera_name not in target_camera_names:
            continue

        if image_camera_name not in parsed_frame:
            parsed_frame[image_camera_name] = dict()
        parsed_frame[image_camera_name]['image'] = parse_image_from_buffer(image.image)
        parsed_frame[image_camera_name]['frame_id'] = frame_id

    # extract the bounding boxes for each camera
    for camera_label in frame.camera_labels:
        label_camera_name = camera_label.name
        if label_camera_name not in parsed_frame:
            continue
        else:
            # Create list of bounding box
            if 'bbox_list' not in parsed_frame[label_camera_name]:
                parsed_frame[label_camera_name]['bbox_list'] = []

            # Extract each bounding box within each camera_label
            for bbox_label in camera_label.labels:
                bbox = convert_bounding_box(bbox_label.box)
                bbox_class = int(bbox_label.type)

                # get object risk
                matched_laser_object_id = waymo_find_best_match_id(frame, bbox_label, use_single_camera)
                if matched_laser_object_id is None:
                    risk = 0
                else:
                    risk = laser_object_risks[matched_laser_object_id]

                # print(str(matched_laser_object_id) + " " + str(risk))
                parsed_frame[label_camera_name]['bbox_list'].append({'class': bbox_class, 'value': bbox, 'risk': risk})

    return parsed_frame


def extract_frame_list(input_file, use_single_camera=False, load_one_frame=False):
    '''
    Extract frame list from the given video file.
    NOTE: data.numpy() requires TF eager execution. The .numpy() method explicitly converts a Tensor to a numpy array.
    '''
    video_segment = tf.data.TFRecordDataset(input_file, compression_type='')

    frame_list = []
    count = 0
    # start = time.time()
    for data in video_segment:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        count += 1
        parsed_frame = extract_image_and_label_from_frame(frame, use_single_camera)
        frame_list.append(parsed_frame)

        if load_one_frame:
            break
    # end = time.time()

    # print("------------------------------------------------------------------------")
    # print("Average time per frame: %f s" % ((end - start) / count))

    return frame_list
