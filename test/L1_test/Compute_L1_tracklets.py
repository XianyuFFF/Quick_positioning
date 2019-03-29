import h5py
import os
import numpy as np
import scipy.io as sio
import pickle

from L1_tracklets import get_valid_detections, create_tracklets
from duke_utils import load_detections_openpose_json
from config_default import configs


dataset_path = configs['dataset_path']
detections_path = os.path.join(dataset_path, 'detections', 'OpenPose')
file_name = configs['file_name']


def compute_L1_tracklets(features, detections, start_frame, end_frame):
    frame_index = 1
    params = configs['tracklets']
    # features = h5py.File(os.path.join(dataset_path, file_name), 'r')['emb']
    # detections = load_detections_openpose_json(detections_path)

    all_dets = detections
    all_det_frames = all_dets[frame_index, :]
    tracklets = []

    for window_start_frame in range(start_frame, end_frame + 1, params['window_width']):
        window_end_frame = window_start_frame + params['window_width']

        window_inds = np.where(np.logical_and(all_det_frames < window_end_frame, all_det_frames >= window_start_frame))[0]

        detections_in_window = np.copy(all_dets[:, window_inds]).T

        detections_conf = np.sum(detections_in_window[:, frame_index + 3:np.size(detections_in_window, axis=1)+1:3], axis=1, dtype=np.float64)
        num_visiable = np.sum(detections_in_window[:, frame_index + 3:np.size(detections_in_window, axis=1)+1:3] > configs['render_threshold'], axis=1, dtype=np.float64)

        vaild = get_valid_detections(detections_in_window, detections_conf, num_visiable, frame_index)
        vaild = np.nonzero(vaild)[0]

        detections_in_window = detections_in_window[vaild, :]

        detections_in_window = np.delete(detections_in_window,
                                         list(range(frame_index + 5, np.size(detections_in_window, 1))), axis=1)
        filtered_detections = detections_in_window

        filtered_features = features[window_inds[vaild], :]
        print(filtered_features)

        create_tracklets(configs, filtered_detections, filtered_features,
                         window_start_frame, window_end_frame, frame_index, tracklets)

    return tracklets


if __name__ == '__main__':
    start_frame = 122178
    end_frame = 181998

    # features = hs.load('test_data/features1.h5', SI_dtype=np.float64)
    features = h5py.File('test_data/features1.h5', 'r')['emb']
    detections = h5py.File('test_data/camera1.mat', 'r')['detections']

    tracklets = compute_L1_tracklets(features, detections, start_frame, end_frame)

    with open('tracklets', 'ab') as dbfile:
        pickle.dump(tracklets, dbfile)

