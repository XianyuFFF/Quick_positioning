import numpy as np
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
from collections import Counter
from itertools import compress
import time

from utils import get_appearance_matrix, get_space_time_affinity
from L1_tracklets.KernighanLin import KernighanLin


def recompute_trajectories(trajectories):
    segment_length = 50

    for i, trajectory in enumerate(trajectories):
        segment_start = trajectory.segment_start
        segment_end = trajectory.segment_end

        num_segments = int((segment_end + 1 - segment_start) / segment_length)
        all_data = np.asarray([np.asarray(tracklet.data) for tracklet in trajectory.tracklets])[0]
        all_data = all_data[all_data[:, 1].argsort()]

        rows = np.unique(all_data[:, 1], return_index=True)[1]
        all_data = all_data[rows, :]
        data_frames = all_data[:, 1]
        interesting_frames = np.round(np.hstack((min(data_frames), np.arange( segment_start + segment_length/2,
                                                                              segment_end, segment_length),
                                                 max(data_frames))))

        key_data = all_data[list(map(lambda x: x in interesting_frames, data_frames)), :]
        key_data[:, 0] = -1
        new_data = fill_trajectories(key_data)

        new_trajectory = trajectories[i]
        sample_tracklet = trajectory.tracklets[0]
        new_trajectory.tracklets = []

        for k in range(num_segments):
            tracklet = sample_tracklet
            tracklet.segment_start = segment_start + k * segment_length
            tracklet.segment_end = tracklet.segment_start + segment_length - 1

            tracklet_frames = np.arange(tracklet.segment_start, tracklet.segment_end)

            tracklet.data = new_data[list(map(lambda x:x in tracklet_frames, new_data[:, 1])), :]

            tracklet.start = min(tracklet.data[:, 0])
            tracklet.finish = max(tracklet.data[:, -1])

            new_trajectory.start_frame = min(new_trajectory.start_frame, tracklet.start)
            new_trajectory.end_frame = max(new_trajectory.end_frame, tracklet.finish)

            new_trajectory.tracklets.append(tracklet)

        trajectories[i] = new_trajectory

    return trajectories


def trajectories_to_top(trajectories):
    data = []
    for i, trajectory in enumerate(trajectories):
        for k, tracklet in enumerate(trajectory.tracklets):
            new_data = tracklet.data
            new_data[:, 1] = i
            data.append(new_data)
    return data


def remove_short_tracks(detections, cutoff_length):
    # This function removes short tracks that have not been associated with
    # any trajectory. Those are likely to be false positives.
    detections_updated = detections
    detections = detections[detections[:, [0, 1]].argsort(),]
    person_ids = np.unique(detections[:, 0])
    lengths = np.histogram(detections[:, 0], person_ids)

    a = person_ids * (lengths < cutoff_length)
    removed_ids = np.extract(a != 0, a)
    np.delete(detections_updated, np.array([person_id for person_id in detections_updated[:, 0]
                                            if person_id in removed_ids]))
    return detections_updated


def fill_trajectories(detections):
    ind = np.lexsort((detections[:, 1], detections[:, 2], detections[:, 3], detections[:, 4], detections[:, 5]))
    detections_updated = np.copy(detections[ind])
    person_ids = np.unique(detections[:, 0])

    for i, person_id in enumerate(person_ids):
        relevant_detections = detections[detections[:, 0] == person_id, :]
        start_frame = np.min(relevant_detections[:, 1])
        end_frame = np.max(relevant_detections[:, 1])

        missing_frames = np.setdiff1d(np.arange(start_frame, end_frame), relevant_detections[:, 1])
        if np.size(missing_frames) == 0:
            continue
        frame_diff = np.diff(missing_frames, n=1, axis=0) > 1
        start_ind = np.hstack((1, frame_diff))
        end_ind = np.hstack((frame_diff, 1))

        start_ind = np.nonzero(start_ind)[0]
        end_ind = np.nonzero(end_ind)[0]

        for k in range(len(start_ind)):
            inter_polated_detections = np.zeros(
                (int(missing_frames[end_ind[k]] - missing_frames[start_ind[k]] + 1), np.size(detections, 1)))

            inter_polated_detections[:, 0] = person_id
            inter_polated_detections[:, 1] = np.arange(missing_frames[start_ind[k]], missing_frames[end_ind[k]]+1)

            pre_detection = detections[(detections[:, 0] == person_id) * detections[:, 1]
                                       == missing_frames[start_ind[k]] - 1, :]
            post_detection = detections[(detections[:, 0] == person_id) * detections[:, 1]
                                        == missing_frames[end_ind[k]] + 1, :]

            for c in range(2, np.size(detections, 1)):
                inter_polated_detections[:, c] = np.linspace(pre_detection[0, c], post_detection[0, c],
                                                             np.size(inter_polated_detections, 0))
            detections_updated = np.vstack((detections_updated, inter_polated_detections))

    return detections_updated


def solve_in_groups(configs, tracklets, labels):
    params = configs["trajectories"]
    if len(tracklets) < params["apperance_groups"]:
        configs['apperance_groups'] = 1

    feature_vectors = np.array([tracklet.feature for tracklet in tracklets], dtype=np.float64)
    feature_vectors = np.reshape(feature_vectors, (len(tracklets), 128))

    # adaptive number of appearance groups
    if params["apperance_groups"] == 0:
        # Increase number of groups until no group is too large to solve
        while True:
            params["apperance_groups"] += 1
            apperance_feature, appearance_groups, _ = k_means(feature_vectors, n_clusters=params["apperance_groups"],
                                                              n_jobs=-1)
            uid, freq = list(zip(*Counter(appearance_groups).items()))
            largest_group_size = max(freq)
            # The BIP solver might run out of memory for large graphs
            if largest_group_size <= 150:
                break
    else:
        # fixed number of appearance groups
        apperance_feature, appearance_groups, _ = k_means(feature_vectors, n_clusters=params["apperance_groups"],
                                                          n_jobs=-1)
    # solve separately for each appearance group
    all_groups = np.unique(appearance_groups)

    result_appearance = []
    for i, group in enumerate(all_groups):
        print("merging tracklets in appearance group {}\n".format(i))
        indices = np.nonzero(appearance_groups == group)[0]
        labels_ = np.asarray(labels)[indices].reshape(-1, 1)
        same_labels = cdist(labels_, labels_) == 0

        # compute appearance and spacetime scores
        appearance_affinity = get_appearance_matrix(feature_vectors[indices], params["threshold"])
        spacetime_affinity, impossibility_matrix, indifference_matrix = get_space_time_affinity([tracklets[i] for
                                                                                                 i in indices],
                                                                                                params["beta"],
                                                                                                params["speed_limit"],
                                                                                                params[
                                                                                                    "indifferent_time"])
        # compute the correlation matrix
        correlation_matrix = appearance_affinity + spacetime_affinity - 1
        correlation_matrix = correlation_matrix * indifference_matrix

        correlation_matrix[impossibility_matrix == 1] = -np.inf
        correlation_matrix[same_labels] = 1

        # just use KL optimizer now
        labels = KernighanLin(correlation_matrix)
        result_appearance.append({"labels": labels, "observations": indices})

    result = {"labels": [], "observations": []}

    for i in range(np.size(np.unique(appearance_groups))):
        merge_results(result, result_appearance[i])

    sorted_result = np.asarray(sorted(zip(result["labels"], result["observations"]), key=lambda x: x[1]))

    result["observations"] = sorted_result[:, 1]
    result["labels"] = sorted_result[:, 0]

    return result


def merge_results(result1, result2):
    if np.size(result1["labels"]) == 0:
        maxinum_label = 0
    else:
        maxinum_label = np.max(result1["labels"])

    result1["labels"].extend(maxinum_label + result2["labels"])
    result1["observations"].extend(result2["observations"])


def find_trajectories_in_window(input_trajectories, start_time, end_time):
    trajecotry_start_frame = np.array([input_trajectory.start_frame for input_trajectory in input_trajectories],
                                      dtype=np.int)
    trajecotry_end_frame = np.array([input_trajectory.end_frame for input_trajectory in input_trajectories],
                                    dtype=np.int)
    trajectories_ind = np.where(np.logical_and(trajecotry_end_frame >= start_time, trajecotry_start_frame <= end_time))[
        0]
    return trajectories_ind


class Trajectory:
    def __init__(self, start_frame, end_frame, segment_start, segment_end):
        self.tracklets = []
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.feature = None


def tracklets_to_trajectory(tracklets, labels):
    unique_labels = np.unique(labels)
    trajectories = []
    for i, label in enumerate(unique_labels):
        trajectory = Trajectory(np.inf, -np.inf, np.inf, -np.inf)
        tracklet_indices = np.nonzero(labels == label)[0]
        for j, ind in enumerate(tracklet_indices):
            trajectory.tracklets.append(tracklets[ind])
            trajectory.start_frame = min(trajectory.start_frame, tracklets[ind].start)
            trajectory.end_frame = max(trajectory.end_frame, tracklets[ind].finish)
            trajectory.segment_start = min(trajectory.segment_start, tracklets[ind].segment_start)
            trajectory.segment_end = max(trajectory.segment_end, tracklets[ind].segment_end)
            trajectory.feature = tracklets[ind].feature
        trajectories.append(trajectory)
    return trajectories


def create_trajectories(configs, input_trajectories, start_frame, end_frame):
    current_trajectories_ind = find_trajectories_in_window(input_trajectories, start_frame, end_frame)
    current_trajectories = [input_trajectories[ind] for ind in current_trajectories_ind]

    if len(current_trajectories) <= 1:
        out_trajectories = input_trajectories
        return out_trajectories

    # select tracklets that will be selected in association.For previously
    # computed trajectories we select only the last three tracklets.
    is_assocations = []
    tracklets = []
    tracklet_labels = []
    for i, current_trajectory in enumerate(current_trajectories):
        for j, tracklet in enumerate(current_trajectory.tracklets):
            tracklets.append(tracklet)
            tracklet_labels.append(i)

            if j >= len(current_trajectory.tracklets) - 4:
                is_assocations.append(True)
            else:
                is_assocations.append(False)

    # solve the graph partitioning problem for each appearance group
    result = solve_in_groups(configs,
                             list(compress(tracklets, is_assocations)),
                             np.array(list(compress(tracklet_labels, is_assocations)))
                             )

    # merge back solution. Tracklets that were associated are now merged back
    # with the rest of tracklets that were sharing the same trajectory
    labels = tracklet_labels
    # labels[is_assocations] = result["labels"]

    count = 0
    for i, is_assocation in enumerate(is_assocations):
        if is_assocation:
            labels[tracklet_labels == tracklet_labels[i]] = result["labels"][count]
            count += 1

    # merge co-identified tracklets to extended tracklets
    new_trajectories = tracklets_to_trajectory(tracklets, labels)

    smooth_trajectories = recompute_trajectories(new_trajectories)

    output_trajectories = input_trajectories
    np.delete(output_trajectories, current_trajectories_ind)

    output_trajectories.extend(smooth_trajectories)

    # show merged tracklets in window
    # TODO visualize

    return output_trajectories































