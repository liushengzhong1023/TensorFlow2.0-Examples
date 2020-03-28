import numpy as np
import core.utils as utils

'''
Postprocessing for partial frames. Convert the partial predictions to original frames.
'''

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
    bboxes = np.array(utils.nms(bboxes, 0.45, method='nms'))

    # ensemble prediction item
    prediction_bbox_item = {'frame_id': frame_meta['frame_id'],
                            'camera_name': frame_meta['camera_name'],
                            'partial_frame_offset': frame_meta['partial_frame_offset'],
                            'bboxes': bboxes}

    return prediction_bbox_item


def map_partial_bbox_to_global(bboxes, partial_frame_offset):
    '''
    Map partial box predictions back to its original frame.
    '''
    mapped_bboxes = np.copy(bboxes)

    for bbox in bboxes:
        if len(bbox) == 0:
            continue
        else:
            mapped_bboxes[:, [0, 2]] += partial_frame_offset[0]
            mapped_bboxes[:, [1, 3]] += partial_frame_offset[1]

    return mapped_bboxes


def merge_partial_pred_bbox(pred_bbox_list):
    '''
    Merge and map all partial pred bbox.
    Output: dict(), key is the camera name, value is their corresponding bbox prediction list.
    bbox: [min_x, min_y, max_x, max_y, conf, class]
    '''
    frame_prediction = {}

    for prediction_bbox_item in pred_bbox_list:
        frame_id = prediction_bbox_item['frame_id']
        camera_name = prediction_bbox_item['camera_name']
        bboxes = prediction_bbox_item['bboxes']
        partial_frame_offset = prediction_bbox_item['partial_frame_offset']

        # map partial bboxes to global bboxes
        mapped_bboxes = map_partial_bbox_to_global(bboxes, partial_frame_offset)

        if camera_name not in frame_prediction:
            frame_prediction[camera_name] = []

        frame_prediction[camera_name].extend(mapped_bboxes)
    print(frame_prediction)

    return frame_prediction
