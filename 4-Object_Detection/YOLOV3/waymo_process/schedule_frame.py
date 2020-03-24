import numpy as np



'''
This file provide different scheduling strategies for frames.
Supported policies:
    1) Serialized full frames
    2) Serialized partial frames
    3) Batched partial frames
'''


def serialize_full_frames(extracted_frame):
    '''
    Serialized full frames.
    '''
    pass


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