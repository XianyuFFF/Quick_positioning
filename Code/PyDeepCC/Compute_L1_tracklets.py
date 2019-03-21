import h5py
import os
import numpy as np

from L1_tracklets import get_valid_detections, create_tracklets
from duke_utils import load_detections_openpose_json
from config_default import configs

os.chdir(os.path.join(os.getcwd(), 'triplet_reid'))
dataset_path = configs['dataset_path']
detections_path = os.path.join(dataset_path, 'detections', 'OpenPose')
file_name = configs['file_name']


def compute_L1_tracklets(total_frame):
    frame_index = 0
    params = configs['tracklets']
    features = h5py.File(os.path.join(dataset_path, file_name), 'r')['emb']
    print(features)
    detections = load_detections_openpose_json(detections_path)
    all_dets = detections

    tracklets = []

    for window_start_frame in range(0, total_frame, params['window_width']):
        window_end_frame = window_start_frame + params['window_width']
        window_frames = list(range(window_start_frame, window_end_frame))

        window_inds = np.where(list(map(lambda x: x in window_frames, all_dets[:, frame_index])))[0]

        detections_in_window = all_dets[window_inds]

        detections_conf = np.sum(detections_in_window[:, frame_index + 3:-1:3], 1)
        num_visiable = np.sum(detections_in_window[:, frame_index + 3:-1:3] > configs['render_threshold'], 1)

        vaild = get_valid_detections(detections_in_window, detections_conf, num_visiable)
        vaild = np.where(vaild)[0]

        detections_in_window = detections_in_window[vaild, :]

        detections_in_window = np.delete(detections_in_window, list(range(5, np.size(detections_in_window, 1))), axis=1)
        filtered_detections = detections_in_window

        filtered_features = features[window_inds[vaild], :]

        tracklets = create_tracklets(configs, filtered_detections, filtered_features,
                                     window_start_frame, window_end_frame)




if __name__ == '__main__':
    compute_L1_tracklets(configs['total_frame'])
