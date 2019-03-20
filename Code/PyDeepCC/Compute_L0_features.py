from config_default import configs
import os
import glob
import scipy.io as sio
import h5py
import numpy as np
from functools import partial
from triplet_reid.embed_detections import embed_detections
from triplet_reid.duke_utils import detections_generator_from_openpose_json, num_detections_from_openpose_json

os.chdir(os.path.join(os.getcwd(), 'triplet_reid'))

print(os.getcwd())

dataset_path = configs['dataset_path']
detections_path = os.path.join(dataset_path, 'detections', 'OpenPose')
# print(detections_path)
if not os.path.exists(detections_path):
    os.makedirs(detections_path, exist_ok=True)

render_threshold = configs['render_threshold']
width = configs['video_width']
height = configs['video_height']

net_config = configs['net']
experiment_root = net_config['experiment_root']
file_name = os.path.join(dataset_path,configs['file_name'])

video_name = "view-Contour2.mp4"

num_detections = num_detections_from_openpose_json(detections_path)
# print(num_detections)
detection_generator = partial(detections_generator_from_openpose_json, video_name, dataset_path, detections_path)
embed_detections(experiment_root, detection_generator, num_detections,  file_name)