import math
import time
import random
import numpy as np
import core.utils as utils

from waymo_process.PriorityQueue import PriorityQueue

'''
This file provide different scheduling strategies for images within ***single*** frame.
Supported policies:
    1) Serialized full frames
    2) Serialized partial frames
    3) Batched partial frames
'''


def serialize_full_frames(extracted_frame):
    '''
    Serialized full frames. We do not consider any bounding box here.
    '''
    image_queue = []
    meta_queue = []

    for camera_name in extracted_frame:
        image = extracted_frame[camera_name]['image']
        preprocessed_image, original_size, new_size = utils.image_preprocess(np.copy(image))

        # no batching
        preprocessed_image = preprocessed_image[np.newaxis, ...]
        image_queue.append(preprocessed_image)

        # get meta information
        image_id = extracted_frame[camera_name]['image_id']
        item = {'image': preprocessed_image,
                'original_size': original_size,
                'new_size': new_size,
                'camera_name': camera_name,
                'image_id': image_id}
        meta_queue.append(item)

    return image_queue, meta_queue


def serialize_partial_frames(extracted_frame):
    '''
    Serialized partial frames. Specifically, each bounding box is processed sequentially.
    '''
    image_queue = []

    for camera_name in extracted_frame:

        image = extracted_frame[camera_name]['image']
        bbox_list = extracted_frame[camera_name]['bbox_list']
        for bbox in bbox_list:
            min_x, min_y, max_x, max_y = bbox['value']
            min_x = math.floor(min_x)
            min_y = math.floor(min_y)
            max_x = math.ceil(max_x)
            max_y = math.ceil(max_y)

            # extract partial image
            partial_image = image[min_y:max_y, min_x:max_x, :]
            preprocessed_partial_image = utils.image_preprocess(np.copy(partial_image))

            # no batching
            preprocessed_partial_image = preprocessed_partial_image[np.newaxis, ...]
            image_queue.append(preprocessed_partial_image)

    return image_queue


def prioritize_serialize_partial_frames(extracted_frame, with_random_boarder=False):
    '''
    Use risk level as priority to schedule the paritial frames in serialized order.
    '''
    image_queue = PriorityQueue()

    for camera_name in extracted_frame:
        image_id = extracted_frame[camera_name]['image_id']
        image = extracted_frame[camera_name]['image']
        bbox_list = extracted_frame[camera_name]['bbox_list']

        # original image size
        original_max_x = image.shape[1]
        original_max_y = image.shape[0]

        for bbox in bbox_list:
            min_x, min_y, max_x, max_y = bbox['value']
            object_id = bbox['object_id']

            min_x = math.floor(min_x)
            min_y = math.floor(min_y)
            max_x = math.ceil(max_x)
            max_y = math.ceil(max_y)

            # add random boarder when extracting the sub frames
            if with_random_boarder:
                random_board_len = random.randint(0, 9)
                min_x = max(0, min_x - random_board_len)
                min_y = max(0, min_y - random_board_len)
                max_x = min(original_max_x, max_x + random_board_len)
                max_y = min(original_max_y, max_y + random_board_len)

            # extract partial image
            partial_image = image[min_y:max_y, min_x:max_x, :]
            preprocessed_partial_image, original_size, new_size = utils.image_preprocess(np.copy(partial_image))
            partial_frame_offset = [min_x, min_y]

            # no batching
            preprocessed_partial_image = preprocessed_partial_image[np.newaxis, ...]

            # ensemble the item
            item = {'image': preprocessed_partial_image,
                    'original_size': original_size,
                    'new_size': new_size,
                    'partial_frame_offset': partial_frame_offset,
                    'camera_name': camera_name,
                    'image_id': image_id,
                    'object_id': object_id}

            # enque
            image_queue.push(item, priority=bbox['risk'])

    output_meta_queue = image_queue.pop_all_item_list()
    output_image_queue = [item['image'] for item in output_meta_queue]

    return output_image_queue, output_meta_queue


def batched_partial_frames(extracted_frame):
    '''
    Batched partial frames. Specifically, bounding boxes with similar sizes are unified and batched.
    TODO: How to preset the best image size for each batch?
    '''
    image_queue = []

    # initialize frame batches
    frame_batches = dict()
    frame_batches[(160, 96)] = []
    frame_batches[(256, 160)] = []
    frame_batches[(480, 320)] = []

    for camera_name in extracted_frame:
        image = extracted_frame[camera_name]['image']
        bbox_list = extracted_frame[camera_name]['bbox_list']
        for bbox in bbox_list:
            min_x, min_y, max_x, max_y = bbox['value']
            min_x = math.floor(min_x)
            min_y = math.floor(min_y)
            max_x = math.ceil(max_x)
            max_y = math.ceil(max_y)

            # extract partial image
            partial_image = image[min_y:max_y, min_x:max_x, :]
            width = max_x - min_x
            height = max_y - min_y

            # decide new size
            if width <= 192 and height <= 128:
                new_size = (160, 96)
            elif width >= 384 and height >= 256:
                new_size = (480, 320)
            else:
                new_size = (256, 160)

            preprocessed_partial_image = utils.image_preprocess(np.copy(partial_image), target_size=new_size)
            frame_batches[new_size].append(preprocessed_partial_image)

    # batching
    for new_size in frame_batches:
        batch = np.array(frame_batches[new_size])
        # print(np.shape(batch))
        image_queue.append(batch)
    # print()

    return image_queue
