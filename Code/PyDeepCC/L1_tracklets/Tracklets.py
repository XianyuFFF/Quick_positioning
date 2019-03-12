from utils import get_bounding_box_centers, estimated_velocities
import numpy as np


def create_tracklets(configs, original_detections, all_features, start_frame, end_frame):
    current_detections_IDX = np.asarray(list(filter(lambda x: start_frame <= x < end_frame, original_detections[:, 1])),
                                        dtype=np.int32)

    if len(current_detections_IDX) < 2:
        return

    detections_centers = get_bounding_box_centers(original_detections[current_detections_IDX, 2:6])
    detection_frames = original_detections[current_detections_IDX, 0]
    estimed_velocity = estimated_velocities(original_detections, start_frame, end_frame,
                                            configs['tracklets']['nearest_neighbors'],
                                            configs['tracklets']['speed_limit'])
