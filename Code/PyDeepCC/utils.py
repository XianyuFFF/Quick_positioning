import numpy as np


def get_bounding_box_centers(bounding_boxs):
    centers = np.hstack(
        (bounding_boxs[:, 0] + 0.5 * bounding_boxs[:, 2],
         bounding_boxs[:, 1] + 0.5 * bounding_boxs[:, 3]))
    return centers


def estimated_velocities(original_detections, start_frame, end_frame, nearest_neighbors, speed_limit):
    search_range_mask = np.asarray(list(filter(lambda x: start_frame - nearest_neighbors <= x <
                                                         end_frame + nearest_neighbors, original_detections)),
                                   dtype=np.int32)
    search_range_centers = get_bounding_box_centers(original_detections[search_range_mask, 2:6])
    search_range_frames = original_detections[search_range_mask, 0]
    detection_indices = np.asarray(list(filter(lambda x: start_frame <= x < end_frame, search_range_frames)),
                                   dtype=np.int32)
