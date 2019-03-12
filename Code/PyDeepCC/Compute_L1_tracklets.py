from config_default import configs
import h5py

def compute_L1_tracklets():
    for i in range(1, 9):
        configs.current_camera = i

        features = None
        detections = None

        tracklets = []

        
