from utils import get_bounding_box_centers, estimated_velocities, get_spatial_group_id, get_appearance_sub_matrix, \
    motion_affinity
import numpy as np
from operator import attrgetter
from scipy.spatial.distance import cdist
from L1_tracklets.KernighanLin import KernighanLin


class SmoothedTracklet:
    def __init__(self, feature, center, center_world, data, features, real_data, mask, start, finish, interval,
                 segment_start, segment_interval, segment_end):
        self.feature = feature
        self.center = center
        self.center_world = center_world
        self.data = data
        self.features = features
        self.real_data = real_data
        self.mask = mask
        self.start = start
        self.finish = finish
        self.interval = interval
        self.segment_start = segment_start
        self.segment_interval = segment_interval
        self.segment_end = segment_end


def smooth_tracklets(tracklets, segement_start, segment_interval, feature_apperance, min_tracklet_length,
                     current_interval, frame_index):
    tracklet_ids = np.unique(tracklets[:, frame_index - 1])
    num_tracklets = len(tracklet_ids)

    smoothed_tracklets = []
    for i in range(num_tracklets):
        mask = tracklets[:, frame_index - 1] == tracklet_ids[i]
        detections = tracklets[mask, :]

        # Reject tracklets of short length
        start = min(detections[:, frame_index])
        finish = max(detections[:, frame_index])

        if (np.size(detections, 0) < min_tracklet_length) or finish - start < min_tracklet_length:
            continue

        interval_length = int(finish - start + 1)
        datapoints = np.arange(start, finish, interval_length)
        frames = detections[:, frame_index]

        current_tracklet = np.zeros((interval_length, np.size(tracklets, 1)))
        current_tracklet[:, frame_index - 1] = np.ones(interval_length) * tracklet_ids[i]
        current_tracklet[:, frame_index] = np.array(start, finish)

        for k in range(frame_index + 1, np.size(tracklets, 1)):
            points = detections[:, k]
            p = np.polyfit(frames, points, 1)
            new_points = np.poly1d(p)(datapoints)
            current_tracklet[:, k] = new_points.T

        median_feature = np.median(feature_apperance[mask], axis=0)
        centers = get_bounding_box_centers(current_tracklet[:, frame_index + 1:frame_index + 5])
        center_point = np.median(centers, axis=0)
        center_point_world = 1

        smoothed_tracklet = SmoothedTracklet(median_feature, center_point, center_point_world, current_tracklet,
                                             feature_apperance, detections, mask, start, finish, current_interval,
                                             segement_start, segment_interval, segment_interval + segement_start - 1)

        smoothed_tracklets.append(smoothed_tracklet)
    return smoothed_tracklets


def create_tracklets(configs, original_detections, all_features, start_frame, end_frame, frame_index, tracklets):
    original_detections_frames = original_detections[:, frame_index]
    current_detections_idx = np.where(
        np.logical_and(original_detections_frames
                       >=
                       start_frame, original_detections_frames < end_frame)
    )[0]

    params = configs['tracklets']
    if len(current_detections_idx) < 2:
        return

    total_labels = 0
    current_interval = 0
    detections_centers = get_bounding_box_centers(
        original_detections[current_detections_idx, frame_index + 1:frame_index + 5])
    detection_frames = original_detections[current_detections_idx, frame_index]

    estimated_velocity = estimated_velocities(original_detections, start_frame, end_frame,
                                              params['nearest_neighbors'],
                                              params['speed_limit'], frame_index)

    spatial_group_ids = get_spatial_group_id(configs['use_groupping'], current_detections_idx, detections_centers,
                                             params)

    print('Creating tracklets: solving space-time groups')

    for spatial_group_ID in range(1, max(spatial_group_ids) + 1):
        elements = np.nonzero(spatial_group_ids == spatial_group_ID)[0]
        spatial_group_observations = current_detections_idx[elements]

        # Create an appearance affinity matrix and motion affinity matrix
        appearance_correlation = get_appearance_sub_matrix(spatial_group_observations, all_features,
                                                           params['threshold'])

        spatial_group_detection_centers = detections_centers[elements, :]
        spatial_group_detection_frames = detection_frames[elements]
        spatial_group_detection_frames = np.reshape(spatial_group_detection_frames,
                                                    (np.size(spatial_group_detection_frames, axis=0), 1))
        spatial_group_estimated_velocity = estimated_velocity[elements, :]
        motion_correlation, imp_matrix = motion_affinity(spatial_group_detection_centers,
                                                         spatial_group_detection_frames,
                                                         spatial_group_estimated_velocity,
                                                         params['speed_limit'], params['beta'])

        # Combine affinities into correlations
        interval_distance = cdist(spatial_group_detection_frames, spatial_group_detection_frames)
        discount_matrix = np.minimum(1, -np.log(interval_distance / params['window_width']))
        correlation_matrix = motion_correlation * discount_matrix + appearance_correlation
        correlation_matrix[imp_matrix == 1] = -np.inf

        print(spatial_group_ID)

        labels = KernighanLin(correlation_matrix)
        labels = labels + total_labels
        total_labels = max(labels)
        identities = labels

        original_detections[spatial_group_observations, frame_index - 1] = identities

    tracklet_to_smooth = original_detections[current_detections_idx, :]
    feature_apperance = all_features[current_detections_idx]
    smoothed_tracklets = smooth_tracklets(tracklet_to_smooth, start_frame, params['window_width'], feature_apperance,
                                          params['min_length'], current_interval, frame_index)

    for i, smoothed_tracklet in enumerate(smoothed_tracklets):
        setattr(smoothed_tracklet, 'id', i)
        setattr(smoothed_tracklet, 'ids', i)

    if smoothed_tracklets:
        tracklets.extend(smoothed_tracklets)

    if tracklets:
        tracklets = sorted(tracklets, key=attrgetter('start', 'finish'))

    return tracklets
