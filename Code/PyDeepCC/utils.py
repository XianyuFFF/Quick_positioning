import numpy as np
import numpy.linalg as LA
from collections import Counter
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster
import cv2


def sigmoid(x, a, c):
    s = 1/(1+np.exp(-a*(x-c)))
    return s


def visiual_tracklets(tracklets, video):
    frame_base_detections = {}
    for i, tracklet in enumerate(tracklets):
        for j, frame in enumerate(range(int(tracklet.start), int(tracklet.finish))):
            if frame_base_detections.get(frame):
                frame_base_detections[frame].append([i, tracklet.data[frame-int(tracklet.start)]])
            else:
                frame_base_detections[frame] = [[i, tracklet.data[frame-int(tracklet.start)]]]

    print(frame_base_detections)

    cap = cv2.VideoCapture(video)

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = frame_base_detections.get(i)
        if detections:
            for k, detection in enumerate(detections):
                id_, det = detection
                _, _, x, y, w, h = det
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (i*17 % 255, i*31 % 255, i*53 % 255))

        cv2.imshow('result', frame)
        cv2.waitKey(100)

        i += 1



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


def get_spatial_group_id(use_grouping, current_detection_idx, detection_centers, params):

    if use_grouping is True:
        pairwise_distance = cdist(detection_centers, detection_centers)
        agglomeration = linkage(pairwise_distance)
        num_spatial_groups = round(params['cluster_coeff'] * len(current_detection_idx) / params['window_width'])
        num_spatial_groups = max(num_spatial_groups, 0)

        while True:
            spatial_group_ids = fcluster(agglomeration, criterion='maxclust', t=num_spatial_groups)
            freq = [num for id_, num in Counter(spatial_group_ids).items()]
            largest_group_size = max(freq)

            # The BIP solver might run out of memory for large graph
            if largest_group_size <= 150:
                return spatial_group_ids
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

    max_required_speed_matrix = np.divide(distance_matrix, np.abs(frame_difference),
                                          out=np.zeros_like(distance_matrix), where=frame_difference != 0)
    prediction_error[max_required_speed_matrix > speed_limit] = np.inf
    impossibility_matrix[max_required_speed_matrix > speed_limit] = 1

    motion_scores = 1 - beta * prediction_error
    return motion_scores, impossibility_matrix


def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


def get_appearance_matrix(feature_vectors, threshold):
    dist = cdist(feature_vectors, feature_vectors)
    appearance_matrix = (threshold - dist) / threshold
    return appearance_matrix


def get_tracklets_features(tracklets, frame_index=1):
    num_tracklets = len(tracklets)
    # bounding box center for each tracklet
    centers_world = {}
    centers_view = {}

    velocity = np.zeros((num_tracklets, 2))
    duration = np.zeros((num_tracklets, 1))
    intervals = np.zeros((num_tracklets, 2))
    startpoint = np.zeros((num_tracklets, 2))
    endpoint = np.zeros((num_tracklets, 2))
    direction = np.zeros((num_tracklets, 2))

    for i, tracklet in enumerate(tracklets):
        detections = tracklet.data
        # 2d points
        bb = detections[:, [2, 3, 4, 5, 1]]
        x = 0.5 * bb[:, 0] + bb[:, 2]
        y = 0.5 * bb[:, 1] + bb[:, 3]
        t = bb[:, 4]
        centers_view[i] = np.vstack((x, y, t)).T
        # 3d points
        centers_world[i] = centers_view[i]
        intervals[i, :] = [t[0], t[-1]]
        startpoint[i, :] = [x[0], y[0]]
        endpoint[i, :] = [x[-1], y[-1]]

        duration[i] = t[-1] - t[0]
        direction[i, :] = endpoint[i, :] - startpoint[i, :]
        velocity[i, :] = direction[i] / duration[i]

    return centers_world, centers_view, startpoint, endpoint, intervals, duration, velocity


def overlap_test(interval1, interval2):
    i1_start, i1_end = interval1
    i2_start, i2_end = interval2
    if max(i1_start, i2_start) <= min(i1_end, i2_end):
        return True
    else:
        return False


def get_space_time_affinity(tracklets, beta, speed_limit, indifference_limit):
    num_tracklets = len(tracklets)
    _, _, startpoint, endpoint, intervals, _, velocity = get_tracklets_features(tracklets)

    center_frame = np.round(np.mean(intervals, axis=1)).reshape((-1, 1))
    frame_difference = cdist(center_frame, center_frame, lambda x, y: x-y)
    overlapping = cdist(intervals, intervals, overlap_test)
    centers = 0.5 * (endpoint + startpoint)
    centers_distance = cdist(centers, centers)
    v = np.logical_or(frame_difference > 0, overlapping)
    merging = np.logical_and(centers_distance < 5, overlapping)

    velocity_x = np.tile(velocity[:, 0], (num_tracklets, 1))
    velocity_y = np.tile(velocity[:, 1], (num_tracklets, 1))

    start_x = np.tile(centers[:, 0], (num_tracklets, 1))
    start_y = np.tile(centers[:, 1], (num_tracklets, 1))
    end_x = np.tile(centers[:, 0], (num_tracklets, 1))
    end_y = np.tile(centers[:, 1], (num_tracklets, 1))

    error_x_forward = end_x + velocity_x * frame_difference - start_x.conj().T
    error_y_forward = end_y + velocity_y * frame_difference - start_y.conj().T

    error_x_backward = start_x.conj().T + velocity_x.conj().T * -frame_difference - end_x
    error_y_backward = start_y.conj().T + velocity_y.conj().T * -frame_difference - end_y

    error_forward = np.sqrt(error_x_forward**2 + error_y_forward**2)
    error_backward = np.sqrt(error_x_backward**2 + error_y_backward**2)

    # check if speed limit is violated
    x_diff = end_x - start_x.conj().T
    y_diff = end_y - start_y.conj().T
    distance_matrix = np.sqrt(x_diff**2 + y_diff**2)
    max_speed_matrix = distance_matrix / np.abs(frame_difference)

    violators = np.asarray(max_speed_matrix > speed_limit)
    violators[np.where(np.logical_not(v))[0]] = 0
    violators = violators + violators.conj().T

    # build impossibility matrix
    impossibility_matrix = np.zeros((num_tracklets, num_tracklets))
    impossibility_matrix[np.logical_and(violators == 1, merging != 1)] = 1
    impossibility_matrix[np.logical_and(overlapping==1, merging != 1)] = 1

    # this is a symmetric matrix, although tracklets are oriented in time
    error_matrix = np.minimum(error_forward, error_backward)
    error_matrix = error_matrix * v
    error_matrix[np.logical_not(v)] = 0
    error_matrix = error_matrix + error_matrix.conj().T
    error_matrix[violators == 1] = np.inf

    # compute indifference matrix
    time_difference = frame_difference * (frame_difference > 0)
    time_difference = time_difference + time_difference.conj().T
    indiff_matrix = 1 - sigmoid(time_difference, 0.1, indifference_limit/2)

    # compute space-time affinities
    st_affinity = 1 - beta * error_matrix
    st_affinity = np.maximum(0, st_affinity)

    return st_affinity, impossibility_matrix, indiff_matrix
