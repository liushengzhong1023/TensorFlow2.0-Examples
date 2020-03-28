import numpy as np
import core.utils as utils

'''
Postprocessing for partial frames. Convert the partial predictions to original frames.
'''


def map_partial_bbox():
    '''
    Map partial box predictions back to its original frame.
    '''
    pass


def postprocess_box_one_batch(pred_bbox, frame_meta):
    '''
    Post process function for partial frame predictions. Attach necessary meta info to the prediction.
    TODO: Currently only support one frame per batch.
    '''
    original_image_size = frame_meta['original_size']
    new_size = frame_meta['new_size']

    # adjust the predicted bounding boxes at current frames
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, new_size, 0.3)

    # non maximum supppression
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    # ensemble prediction item
    prediction_bbox_item = {'frame_id': frame_meta['frame_id'],
                            'camera_name': frame_meta['camera_name'],
                            'partial_frame_offset': frame_meta['partial_frame_offset'],
                            'bboxes': bboxes}

    return prediction_bbox_item