import math
import numpy as np
import core.utils as utils

'''
This file provide different scheduling strategies for frames.
Supported policies:
    1) Serialized full frames
    2) Serialized partial frames
    3) Batched partial frames
'''


def serialize_full_frames(frame_list):
    '''
    Serialized full frames. We do not consider any bounding box here.
    '''
    image_queue = []
    for extracted_frame in frame_list:
        for camera_name in extracted_frame:
            image = extracted_frame[camera_name]['image']
            preprocessed_image = utils.image_preprocess(np.copy(image))

            # no batching
            preprocessed_image = preprocessed_image[np.newaxis, ...]
            image_queue.append(preprocessed_image)

    return image_queue


def serialize_partial_frames(frame_list):
    '''
    Serialized partial frames. Specifically, each bounding box is processed sequentially.
    '''
    image_queue = []
    for extracted_frame in frame_list:
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


def batched_partial_frames(frame_list):
    '''
    Batched partial frames. Specifically, bounding boxes with similar sizes are unified and batched.
    TODO: How to preset the best image size for each batch?
    '''
    image_queue = []
    for extracted_frame in frame_list:
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
