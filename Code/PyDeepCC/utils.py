import numpy as np
import numpy.linalg as LA
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster


def get_bounding_box_centers(bounding_boxs):
    centers = np.vstack(
        (bounding_boxs[:, 0] + 0.5 * bounding_boxs[:, 2],
         bounding_boxs[:, 1] + 0.5 * bounding_boxs[:, 3]))
    return centers.T

def estimated_velocities(original_detections, start_frame, end_frame, nearest_neighbors, speed_limit, frame_index):
    search_range_mask = np.where(np.logical_and(original_detections[:, frame_index] >= start_frame-nearest_neighbors,
                                                original_detections[:, frame_index] < end_frame+nearest_neighbors))[0]
    # due to main code lack camera, so this is [1:5] in main code
    search_range_centers = get_bounding_box_centers(original_detections[search_range_mask, frame_index+1:frame_index+5])
    search_range_frames = np.asarray(original_detections[search_range_mask, frame_index], dtype=int)
    detection_indices = np.where(np.logical_and(search_range_frames >= start_frame, search_range_frames < end_frame))[0]

    # Compute all pairwise distances
    pair_distance = cdist(search_range_centers, search_range_centers)
    num_detections = len(detection_indices)
    estimated_velocities_ = np.zeros((num_detections, 2))

    # Estimate the velocity of each detection
    for i, current_detection_index in enumerate(detection_indices):

        velocities = []
        current_frame = search_range_frames[current_detection_index]

        for frame in range(current_frame - nearest_neighbors, current_frame + nearest_neighbors + 1):
            if abs(current_frame - frame) <= 0:
                continue

            detections_at_this_time_instant = search_range_frames == frame
            if np.sum(detections_at_this_time_instant) == 0:
                continue

            distances_at_this_time_instant = np.copy(pair_distance[current_detection_index, :])
            distances_at_this_time_instant[np.where(np.logical_not(detections_at_this_time_instant))[0]] = np.inf

            # Find detection closest to the current detection
            target_detection_index = np.argmin(distances_at_this_time_instant)
            estimated_distance = (search_range_centers[target_detection_index, :] -
                                  search_range_centers[current_detection_index, :])
            estimated_velocity = estimated_distance / (search_range_frames[target_detection_index] -
                                                       search_range_frames[current_detection_index])

            # Check if speed limit is violated
            estimated_speed = LA.norm(estimated_velocity)
            if estimated_speed > speed_limit:
                continue

            # Update velocity estimates
            velocities.append(estimated_velocity)
        velocities = np.asarray(velocities)
        if velocities.size == 0:
            velocities = np.array([[0, 0]])
        # Estimate the velocity
        estimated_velocities_[i, 0] = np.mean(velocities[:, 0])
        estimated_velocities_[i, 1] = np.mean(velocities[:, 1])

    return estimated_velocities_


def get_spatial_group_id(use_grouping, current_detection_IDX, detection_centers, params):

    if use_grouping is True:
        pairwise_distance = cdist(detection_centers, detection_centers)
        agglomeration = linkage(pairwise_distance)
        num_spatial_groups = round(params['cluster_coeff'] * len(current_detection_IDX) / params['window_width'])
        num_spatial_groups = max(num_spatial_groups, 0)

        while True:
            spatial_group_IDs = fcluster(agglomeration, criterion='maxclust', t=num_spatial_groups)
            uid = np.unique(spatial_group_IDs)
            freq = np.hstack((np.histogram(spatial_group_IDs.flatten()), uid))

            largest_group_size = len(freq)

            # The BIP solver might run out of memory for large graph
            if largest_group_size <= 150:
                return spatial_group_IDs
            num_spatial_groups += 1


def get_appearance_sub_matrix(spatial_group_observations, features_vectors, threshold):
    features = features_vectors[spatial_group_observations]
    dist = cdist(features, features)
    correlation = (threshold - dist) / threshold
    return correlation

def motion_affinity(detection_centers, detection_frames, estimated_velocity, speed_limit, beta):
    # This function compute the motion affinities given a set of detections
    # A simple motion prediction is performed from a source detection to
    # a target detection to compute the prediction error
    num_detections = np.size(detection_centers, 0)
    impossibility_matrix = np.zeros((len(detection_frames), len(detection_frames)))

    frame_difference = cdist(detection_frames, detection_frames)
    velocity_X = np.tile(estimated_velocity[:, 0], (num_detections, 1)).T
    velocity_Y = np.tile(estimated_velocity[:, 1], (num_detections, 1)).T
    center_X = np.tile(detection_centers[:, 0], (num_detections, 1)).T
    center_Y = np.tile(detection_centers[:, 1], (num_detections, 1)).T

    error_X_forward = center_X + velocity_X * frame_difference - center_X.conj().transpose()
    error_Y_forward = center_Y + velocity_Y * frame_difference - center_Y.conj().transpose()

    error_X_backward = center_X.conj().transpose() + velocity_X.conj().transpose() * -frame_difference.conj().transpose() - center_X
    error_Y_backward = center_Y.conj().transpose() + velocity_Y.conj().transpose() * -frame_difference.conj().transpose() - center_Y

    error_forward = np.sqrt(error_X_forward ** 2 + error_Y_forward ** 2)
    error_backward = np.sqrt(error_X_backward**2 + error_Y_backward ** 2)

    # only upper triangular part is valid
    prediction_error = np.minimum(error_forward, error_backward)
    prediction_error = np.triu(prediction_error) + np.triu(prediction_error).conj().T

    # Check if speed limit is violated
    x_diff = center_X - center_X.conj().T
    y_diff = center_Y - center_Y.conj().T
    distance_matrix = np.sqrt(x_diff**2 + y_diff ** 2)

    max_required_speed_matrix = np.divide(distance_matrix, np.abs(frame_difference), out=np.zeros_like(distance_matrix), where=frame_difference!=0)
    prediction_error[max_required_speed_matrix > speed_limit] = np.inf
    impossibility_matrix[max_required_speed_matrix > speed_limit] = 1

    motion_scores = 1 - beta * prediction_error
    return motion_scores, impossibility_matrix

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

