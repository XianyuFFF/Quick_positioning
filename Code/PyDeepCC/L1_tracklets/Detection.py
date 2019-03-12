import numpy as np
from duke_utils import pose2bb, scale_bb, feet_position


def get_valid_detections(detections_in_interval, detection_conf, num_visible, configs, iCam):
    valid = np.ones((np.size(detections_in_interval, 0)))
    for i in range(np.size(detections_in_interval, 0)):
        pose = detections_in_interval[i, 2:]
        bb = pose2bb(pose)
        new_bb, new_pose = scale_bb(bb, pose, 1.25)
        feet = feet_position(new_bb)
        detections_in_interval[i, 2:6] = new_bb

        if new_bb[2] < 20 or new_bb[3] < 20 or new_bb[3] > 450:
            valid[i] = 0
            continue

        if num_visible[i] < 5 or detection_conf[i] < 4:
            valid[i] = 0
            continue

        # current no apply
        # if not inpolygon(feet, configs, iCam):
        #     valid[i] = 0
        #     continue

    return valid, detections_in_interval
