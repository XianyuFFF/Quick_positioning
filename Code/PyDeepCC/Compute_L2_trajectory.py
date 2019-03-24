from Trajectory import create_trajectories, tracklets_to_trajectory
from config_default import configs


def compute_L2_trajectories(configs, tracklets, start_frame, end_frame):
    trajectories_from_tracklets = tracklets_to_trajectory(tracklets, list(range(1, len(tracklets)+1)))
    trajectories = trajectories_from_tracklets
    trajectories = create_trajectories(configs, trajectories, start_frame, end_frame)

    