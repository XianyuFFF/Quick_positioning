from config_default import configs
import h5py
from L1_tracklets.Tracklets import create_tracklets

def compute_L1_tracklets():
    for i in range(1, 9):
        configs.current_camera = i

        features = None
        detections = None

        # tracklets = create_tracklets(configs)
    # TODO