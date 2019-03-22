import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist

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

if __name__ == '__main__':
    beta = 0.02
    speed_limit = 20

    detection_frames = pd.read_csv('detection_frames.txt',sep='\t',header=None).to_numpy()
    detection_centers = pd.read_csv('detection_centers.txt',sep='\t',header=None).to_numpy()
    estimated_velocity = pd.read_csv('estimated_velocity.txt',sep='\t',header=None).to_numpy()

    print(motion_affinity(detection_centers, detection_frames, estimated_velocity, speed_limit, beta))
