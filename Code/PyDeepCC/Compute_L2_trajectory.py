from Trajectory import *
from config_default import configs


def compute_L2_trajectories(configs, tracklets, start_frame, end_frame):
    trajectories_from_tracklets = tracklets_to_trajectory(tracklets, list(range(1, len(tracklets)+1)))
    trajectories = trajectories_from_tracklets

    while start_frame <= end_frame:
        trajectories = create_trajectories(configs, trajectories, start_frame, end_frame)
        start_frame = end_frame - configs['trajectories']['overlap']
        end_frame = start_frame + configs['trajectories']['window_width']

    # Convert trajectories
    tracker_output_raw = trajectories_to_top(trajectories)
    # Interpolate missing detections
    tracker_output_filled = fill_trajectories(tracker_output_raw)
    # Remove spurius tracks
    tracker_output_removed = remove_short_tracks(tracker_output_filled,
                                                 configs['trajectories']['minimum_trajectory_length'])

    _, index = np.unique(tracker_output_removed[:, 0], return_inverse=True)
    tracker_output_removed[:, 0] = index
    tracker_output = tracker_output_removed[tracker_output_removed[:, [0, 1]].argsort(),]
    return tracker_output
