import json
import os
import numpy as np


def extract_time_files(input_path):
    '''
    Extract time file list from the given input path.
    '''
    filename_list = os.listdir(input_path)
    file_list = []

    for file in filename_list:
        if 'time' in file:
            file_list.append(os.path.join(input_path, file))

    return file_list


def compute_mean_latency_for_full(input_file):
    '''
    Compute mean latency per frame for full serialized frames.
    '''
    with open(input_file, "r") as f:
        times = json.load(f)
    frame_times = []

    for segment in times:
        for frame in times[segment]:
            frame_times.append(times[segment][frame])

    mean_latency_per_frame = np.mean(frame_times)

    print("Filename: " + input_file)
    print("Mean latency per frame: %f s" % (mean_latency_per_frame))


def compute_mean_latency_for_partial(input_file):
    '''
    Compute mean latency per frame for partial serialized frames.
    '''
    with open(input_file, "r") as f:
        times = json.load(f)
    frame_times = []

    for segment in times:
        for frame in times[segment]:
            frame_latency = 0
            for object in times[segment][frame]:
                frame_latency += times[segment][frame][object]

            frame_times.append(frame_latency)

    mean_latency_per_frame = np.mean(frame_times)

    print("Filename: " + input_file)
    print("Mean latency per frame: %f s" % (mean_latency_per_frame))


if __name__ == "__main__":
    input_path = "/home/sl29/DeepScheduling/src/TensorFlow2.0-Examples/4-Object_Detection/YOLOV3/data/prediction_results"

    time_file_list = extract_time_files(input_path)

    for time_file in time_file_list:
        if "partial_frames" in time_file:
            compute_mean_latency_for_partial(time_file)
        else:
            compute_mean_latency_for_full(time_file)