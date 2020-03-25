import numpy as np
import math
import cv2
import tensorflow as tf

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

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


def extract_image_and_label_from_frame(frame):
    '''
    Extract the 2d image, and its corresponding labels from the given frame.
    NOTE: 5 cameras deployed (front, front left, front right, side left, side right); JPEG image for each camera.
    output: Dict of 5 dict, where the first level key is image name;
            The second level dict['image']=image, dict['bbox_list']=list of bounding boxes
    '''
    parsed_frame = dict()

    # extract the image for each camera
    for image in frame.images:
        image_camera_name = open_dataset.CameraName.Name.Name(image.name)
        if image_camera_name not in parsed_frame:
            parsed_frame[image_camera_name] = dict()
        parsed_frame[image_camera_name]['image'] = parse_image_from_buffer(image.image)

    # extract the camera label for each camera
    for camera_label in frame.camera_labels:
        label_camera_name = open_dataset.CameraName.Name.Name(camera_label.name)
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
                # object_id = bbox_label.id
                parsed_frame[label_camera_name]['bbox_list'].append({'class': bbox_class, 'value': bbox})

    return parsed_frame


def extract_frame_list(input_file):
    '''
    Extract frame list from the given video file.
    '''
    video_segment = tf.data.TFRecordDataset(input_file, compression_type='')

    frame_list = []
    for data in video_segment:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        parsed_frame = extract_image_and_label_from_frame(frame)
        frame_list.append(parsed_frame)

    return frame_list