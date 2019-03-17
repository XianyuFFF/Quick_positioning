from utils import get_bounding_box_centers, estimated_velocities, get_spatial_group_id, get_appearance_sub_matrix, motion_affinity
import numpy as np
from scipy.spatial.distance import cdist
from L1_tracklets.KernighanLin import KernighanLin


def create_tracklets(configs, original_detections, all_features, start_frame, end_frame):
    current_detections_IDX = np.asarray(list(filter(lambda x: start_frame <= x < end_frame, original_detections[:, 1])),
                                        dtype=np.int32)
    params = configs['tracklets']
    if len(current_detections_IDX) < 2:
        return
    total_labels = 0
    detections_centers = get_bounding_box_centers(original_detections[current_detections_IDX, 2:6])
    detection_frames = original_detections[current_detections_IDX, 0]
    estimated_velocity = estimated_velocities(original_detections, start_frame, end_frame,
                                            params['nearest_neighbors'],
                                            params['speed_limit'])
    spatial_group_IDs = get_spatial_group_id(configs['use_groupping'], current_detections_IDX, detections_centers,
                                             params)

    print('Creating tracklets: solving space-time groups')

    for spatial_group_ID in range(0, max(spatial_group_IDs)):
        elements = np.nonzero(spatial_group_IDs == spatial_group_ID)[0]
        spatial_group_observations = current_detections_IDX[elements]

        # Create an appearance affinity matrix and motion affinity matix
        appearance_correlation = get_appearance_sub_matrix(spatial_group_observations, all_features, params['threshold'])
        spatial_group_detection_centers = detections_centers[elements, :]
        spatial_group_detection_frames = detection_frames[elements, :]
        spatial_group_estimated_velocity = estimated_velocity[elements, :]
        motion_correlation, imp_matrix = motion_affinity(spatial_group_detection_centers,
                                                         spatial_group_detection_frames,
                                                         spatial_group_estimated_velocity,
                                                         params['speed_limit'], params['beta'])

        # Combine affinities into correlations
        interval_distance = cdist(spatial_group_detection_frames, spatial_group_detection_frames)
        discount_matrix = min(1, -np.log(interval_distance/params['window_width']))
        correlation_matrix = motion_correlation * discount_matrix + appearance_correlation
        correlation_matrix[imp_matrix == 1] = -np.inf

        print(spatial_group_ID)

        labels = KernighanLin(correlation_matrix)
        labels = labels + total_labels
        total_labels = max(labels)
        identities = labels

        original_detections[spatial_group_ID, 0] = identities










