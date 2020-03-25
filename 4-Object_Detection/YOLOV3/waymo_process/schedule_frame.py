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

            # no batch
            preprocessed_image = preprocessed_image[np.newaxis, ...]
            image_queue.append(preprocessed_image)

    return image_queue


def serialize_partial_frames(extracted_frame):
    '''
    Serialized partial frames. Specifically, each bounding box is processed sequentially.
    '''
    pass



def batched_partial_frames(extracted_frame):
    '''
    Batched partial frames. Specifically, bounding boxes with similar sizes are unified and batched.
    '''
    pass