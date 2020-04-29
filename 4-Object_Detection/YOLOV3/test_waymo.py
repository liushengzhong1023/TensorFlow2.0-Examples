import cv2
import os
import time
import argparse
import random
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo_process.parse_frame import extract_image_and_label_from_frame, extract_frame_list
from waymo_process.schedule_frame import *
from waymo_process.partial_frame_postprocess import *
from waymo_process.waymo_test_utils import *

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument("-scheduling_policy", type=str,
                    default="prioritize_serialize_partial_frames",
                    help="The choice of scheduling policy.")
parser.add_argument("-add_random_boarder", type=str,
                    default="False",
                    help="Flag about whether to add random boarder around bounding boxes")
parser.add_argument("-GPU", type=str,
                    default="False",
                    help="Flag about wheter to use GPU for inference")
parser.add_argument("-sampling", type=str,
                    default="False",
                    help="Whether to sample the frames, used for time profiling")
args = parser.parse_args()

if args.GPU == "True" or args.GPU == "true":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# set CPU threads number
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
THREADS_NUM = 2 * 2


def test_single_file(model, input_file, scheduling_policy, with_random_boarder=False, sampling=False):
    '''
    The test function for a given input Waymo record.
    '''
    # print log information
    print("Testing: " + input_file)

    # placehoder for prediction result and time result
    segment_predictions = dict()
    segment_times = dict()

    # Extract the whole segment, should have 200 frames
    start = time.time()
    frame_list = extract_frame_list(input_file, use_single_camera=True, load_one_frame=False)
    frame_count = len(frame_list)
    end = time.time()
    print("------------------------------------------------------------------------")
    print("Frame count: " + str(frame_count))
    print("File reading and parsing time: %f s" % (end - start))

    # # ---------------------------------------------Scheduling & Inference---------------------------------------------
    '''
    Each frame is scheduled and predicted in serialized order; the scheduling within each frame is considered.

    '''
    for extracted_frame in frame_list:
        # sampling the frames
        if sampling:
            random_flag = random.random()
            if random_flag > 0.05:
                continue

        # scheduling
        if scheduling_policy == "serialize_full_frames":
            image_queue, meta_queue = serialize_full_frames(extracted_frame)
        elif scheduling_policy == "serialize_partial_frames":
            image_queue = serialize_partial_frames(extracted_frame)
        elif scheduling_policy == "prioritize_serialize_partial_frames":
            image_queue, meta_queue = prioritize_serialize_partial_frames(extracted_frame, with_random_boarder)
        else:
            image_queue = batched_partial_frames(extracted_frame)
        # print("------------------------------------------------------------------------")
        # print('Image batch count: ' + str(len(image_queue))):

        # warm up run for one image
        if image_queue:
            warmup_image = image_queue[0]
            pred_bbox = model.predict(warmup_image)

        # predictions
        pred_bbox_list = []
        for (i, image_batch) in enumerate(image_queue):
            start = time.time()
            pred_bbox = model.predict(image_batch)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            # count time
            end = time.time()
            duration = end - start

            # attach frame meta info to the predicted bbox
            frame_meta = meta_queue[i]
            processed_bboxes = postprocess_box_one_batch(pred_bbox, frame_meta)
            pred_bbox_list.append(processed_bboxes)

            # update time records
            if scheduling_policy == "serialize_full_frames":
                image_id = frame_meta['image_id']
                segment_times[image_id] = duration
            elif scheduling_policy == "prioritize_serialize_partial_frames":
                image_id = frame_meta['image_id']
                object_id = frame_meta['object_id']

                if image_id not in segment_times:
                    segment_times[image_id] = dict()

                segment_times[image_id][object_id] = duration

        # merge and map partial frame pred bbox
        if scheduling_policy == "prioritize_serialize_partial_frames":
            frame_pred_bbox = merge_partial_pred_bbox(pred_bbox_list)
        else:
            frame_pred_bbox = extract_full_pred_bbox(pred_bbox_list)

        # update output
        frame_output = get_frame_output(frame_pred_bbox)
        segment_predictions = merge_frame_predictions(segment_predictions, frame_output)

    return segment_predictions, segment_times


if __name__ == "__main__":
    # -------------------------------------------------- Initialize model ----------------------------------------------
    # NOTE: the shape param does not include the batch size
    input_layer = tf.keras.layers.Input([None, None, 3])
    feature_maps = YOLOv3(input_layer)

    # decode bounding boxes
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(inputs=input_layer, outputs=bbox_tensors)
    utils.load_weights(model,
                       "/home/sl29/DeepScheduling/src/TensorFlow2.0-Examples/4-Object_Detection/YOLOV3/yolov3.weights")

    # -----------------------------------------------Load Waymo data----------------------------------------------------
    '''
    Output is a numpy array of given size: [1, input_size, input_size, 3], batch size can be changed.
    '''
    input_path = "/home/sl29/data/Waymo/validation"
    output_path = "/home/sl29/DeepScheduling/src/TensorFlow2.0-Examples/4-Object_Detection/YOLOV3/data/prediction_results"
    input_files = extract_files(input_path)

    scheduling_policy = args.scheduling_policy
    with_random_boarder = args.add_random_boarder

    if args.GPU == "True" or args.GPU == "true":
        use_GPU = True
    else:
        use_GPU = False

    if args.sampling == "True" or args.sampling == "true":
        sampling_flag = True
    else:
        sampling_flag = False

    # outputfiles
    predictions_file = get_prediction_file_name(output_path, scheduling_policy, with_random_boarder)
    times_file = get_time_file_name(output_path, scheduling_policy, use_GPU, THREADS_NUM)

    # placeholder for all predictions
    predictions = dict()
    times = dict()

    for input_file in input_files:
        segment_predictions, segment_times = test_single_file(model, input_file,
                                                              scheduling_policy,
                                                              with_random_boarder,
                                                              sampling_flag)
        predictions = merge_segment_predictions(predictions, input_file, segment_predictions)
        times = merge_segment_times(times, input_file, segment_times)

    # save predictions to file
    # save_prediction_to_file(predictions, predictions_file)

    # save times to file
    save_time_to_file(times, times_file)
