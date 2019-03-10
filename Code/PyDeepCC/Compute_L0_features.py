from config_default import configs
import os
import glob
import scipy.io as sio
import h5py

dataset_path = configs['dataset_path']
detection_path = os.path.join(dataset_path, 'detections', 'OpenPose')

detection_files = glob.glob(detection_path+'/*.mat')

for i, detection_file in enumerate(detection_files):
    with h5py.File(detection_file, 'r') as poses:
        print(poses['detections'])



# command = 'python3 embed_detections.py --experiment_root {} --dataset_path {} --detections_path {} --filename {}'
#
# os.system()