import numpy as np
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
from collections import Counter

from utils import get_appearance_matrix


def solve_in_groups(configs, tracklets, labels):
    params = configs["trajectories"]
    if len(tracklets) < params["appearance_groups"]:
        configs['appearance_groups'] = 1

    feature_vectors = np.array([tracklet.feature for tracklet in tracklets])
    feature_vectors = np.reshape(feature_vectors, (len(tracklets), 128))

    # adaptive number of appearance groups
    if params["appearance_groups"] == 0:
        # Increase number of groups until no group is too large to solve
        while True:
            params["appearance_groups"] += 1
            apperance_feature, appearance_groups,_ = k_means(feature_vectors, n_clusters=params["appearance_groups"],
                                                             n_jobs=-1)
            uid, freq = list(zip(*Counter(appearance_groups).items()))
            largest_group_size = max(freq)
            # The BIP solver might run out of memory for large graphs
            if largest_group_size <= 150:
                break
    else:
        apperance_feature, appearance_groups, _ = k_means(feature_vectors, n_clusters=params["appearance_groups"],
                                                          n_jobs=-1)
    all_groups = np.unique(appearance_groups)

    for i, group in enumerate(all_groups):
        print("merging tracklets in appearance group {}\n".format(i))
        indices = np.nonzero(appearance_groups == group)
        same_labels = cdist(labels[indices], labels[indices]) == 0

        # compute appearance and spacetime scores
        appearance_affinity = get_appearance_matrix(feature_vectors[indices], params["threshold"])
        spacetime_affinity, impossibility_matrix, indifference_matrix = get_space_time_affinity(tracklets[indices],
                                                                                                params["beta"],
                                                                                                params["speed_limit"],
                                                                                                params["indifference_time"])



















def find_trajectories_in_window(input_trajectories, start_time, end_time):
    trajecotry_start_frame = np.array([input_trajectory.start_frame for input_trajectory in input_trajectories],
                                      dtype=np.int)
    trajecotry_end_frame = np.array([input_trajectory.end_frame for input_trajectory in input_trajectories],
                                      dtype=np.int)
    trajectories_ind = np.where(np.logical_and(trajecotry_end_frame >= start_time, trajecotry_start_frame <= end_time))
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
        tracklet_indices = np.nonzero(labels == label)
        for j, ind in enumerate(tracklet_indices):
            trajectory.tracklets.append(tracklets[ind])
            trajectory.start_frame = min(trajectory.start_frame, tracklets[ind].start)
            trajectory.end_frame = max(trajectory.end_frame, tracklets[ind].finish)
            trajectory.segment_start = min(trajectory.segment_start, tracklets[ind].segment_start)
            trajectory.feature = tracklets[ind].feature
        trajectories.append(trajectory)
    return trajectories


def create_trajectories(configs, input_trajectories, start_frame, end_frame):
    current_trajectories_ind = find_trajectories_in_window(input_trajectories, start_frame, end_frame)
    current_trajectories = input_trajectories[current_trajectories_ind]
    if len(current_trajectories) <=1:
        out_trajectories = input_trajectories
        return out_trajectories

    # select tracklets that will be selected in association.For previously
    # computed trajectories we select only the last three tracklets.
    is_assocation = []
    tracklets = []
    tracklet_labels = []
    for i, current_trajectory in enumerate(current_trajectories):
        for j, tracklet in enumerate(current_trajectory.tracklets):
            tracklets.append(tracklet)
            tracklet_labels.append(i)

            if j >= len(current_trajectory.tracklets)-4:
                is_assocation.append(True)
            else:
                is_assocation.append(False)

    is_assocation = np.array(is_assocation)
    result = solve_in_groups(configs, tracklets[is_assocation], tracklet_labels[is_assocation])


