import numpy as np
import pandas as pd


def get_bounding_box_centers(bounding_boxs):
    centers = np.vstack(
        (bounding_boxs[:, 0] + 0.5 * bounding_boxs[:, 2],
         bounding_boxs[:, 1] + 0.5 * bounding_boxs[:, 3]))
    return centers.T


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
        datapoints = np.linspace(start, finish, interval_length)
        frames = detections[:, frame_index]

        current_tracklet = np.zeros((interval_length, np.size(tracklets, 1)))
        current_tracklet[:, frame_index - 1] = np.ones(interval_length) * tracklet_ids[i]
        current_tracklet[:, frame_index] = np.array(start, finish)

        for k in range(frame_index + 1, np.size(tracklets, 1)):
            points = detections[:, k]
            p = np.polyfit(frames, points, 1)
            new_points = np.poly1d(p)(datapoints)
            current_tracklet[:, k] = new_points.T

        # median_feature = np.median(feature_apperance[mask], axis=0)
        median_feature = None
        centers = get_bounding_box_centers(current_tracklet[:, frame_index + 1:frame_index + 5])
        center_point = np.median(centers, axis=0)
        center_point_world = 1

        smoothed_tracklet = SmoothedTracklet(None, center_point, center_point_world, current_tracklet,
                                             None, detections, mask, start, finish, current_interval,
                                             segement_start, segment_interval, segment_interval + segement_start - 1)

        smoothed_tracklets.append(smoothed_tracklet)
    return smoothed_tracklets


if __name__ == '__main__':
    tracklets_to_smooth = pd.read_csv('tracklets_to_smooth.txt', sep='\t', header=None).to_numpy()
    tracklets_to_smooth[:, [0, 1]] = tracklets_to_smooth[:, [1, 0]]
    feature_apperance = None
    current_interval = 0
    min_tracklet_length = 5
    segment_interval = 50
    segment_start = 122178
    smooth_tracklets(tracklets_to_smooth, segment_start, segment_interval, feature_apperance,
                     min_tracklet_length,current_interval, 1)


