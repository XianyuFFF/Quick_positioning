import numpy as np
from scipy.linalg import qr

def solve_minnonzero(A, b):
    x1, res, rnk, s = np.linalg.lstsq(A, b)
    if rnk == A.shape[1]:
        return x1   # nothing more to do if A is full-rank
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    Z = Q[:, rnk:].conj()
    C = np.linalg.solve(Z[rnk:], -x1[rnk:])
    return x1 + Z.dot(C)


def pose2bb(pose, render_threshold, image_width, image_height):
    # Template pose
    ref_pose = np.array([[0, 0],  # nose
                         [0, 23],  # neck
                         [28, 23],  # rshoulder
                         [39, 66],  # relbow
                         [45, 108],  # rwrist
                         [-28, 23],  # lshoulder
                         [-39, 66],  # lelbow
                         [-45, 108],  # lwrist
                         [20, 106],  # rhip
                         [20, 169],  # rknee
                         [20, 231],  # rankle
                         [-20, 106],  # lhip
                         [-20, 169],  # lknee
                         [-20, 231],  # lanke
                         [5, -7],  # reye
                         [11, -8],  # rear
                         [-5, -7],  # leye
                         [-11, -8],  # lear
                         ], dtype=np.float32)

    ref_bb = np.array([[-50, -15], [50, 240]], dtype=np.float32)

    pose = np.reshape(pose, (18, 3))

    valid = np.logical_and(pose[:, 0] != 0, pose[:, 1] != 0, pose[:, 2] > render_threshold)

    if sum(valid) < 2:
        bb = [0, 0, 0, 0]
        return bb

    points_det = pose[valid, :2]
    points_reference = ref_pose[valid, :]

    # Compute minimum enclosing rectangle
    base_left = min(points_det[:, 0])
    base_top = min(points_det[:, 1])
    base_right = max(points_det[:, 0])
    base_bottom = max(points_det[:, 1])

    # Fit pose to template
    M = np.size(points_det, 0)
    B = np.hstack((points_det[:, 0], points_det[:, 1]))
    A = np.vstack(
        (
            np.hstack(
                (np.reshape(points_reference[:, 0], (M, 1)), np.zeros((M, 1)), np.ones((M, 1)), np.zeros((M, 1)))),
            np.hstack((np.zeros((M, 1)), np.reshape(points_reference[:, 1], (M, 1)), np.zeros((M, 1)), np.ones((M, 1))))
        )
    )

    params = solve_minnonzero(A, B)

    M = 2
    A2 = np.vstack(
        (
            np.hstack((np.reshape(ref_bb[:, 0], (M, 1)), np.zeros((M, 1)), np.ones((M, 1)), np.zeros((M, 1)))),
            np.hstack((np.zeros((M, 1)), np.reshape(ref_bb[:, 1], (M, 1)), np.zeros((M, 1)), np.ones((M, 1))))
        )
    )

    result = A2 @ params

    fit_left = min(result[:2])
    fit_top = min(result[2:])
    fit_right = max(result[:2])
    fit_bottom = max(result[2:])

    # Fuse bounding boxes

    left = min(base_left, fit_left)
    top = min(base_top, fit_top)
    right = max(base_right, fit_right)
    bottom = max(base_bottom, fit_bottom)

    left, right = left * image_width, right * image_width
    top, bottom = top * image_height, bottom * image_height

    h, w = bottom - top + 1, right - left + 1

    bb = np.array([left, top, w, h], dtype=np.float32)

    return bb


if __name__ == '__main__':
    pose = [0.127025000000000,
            0.0989317000000000,
            0.353739000000000,
            0.112869000000000,
            0.118552000000000,
            0.499880000000000,
            0.106039000000000,
            0.115743000000000,
            0.502682000000000,
            0.100784000000000,
            0.143767000000000,
            0.102752000000000,
            0.0992115000000000,
            0.175514000000000,
            0.0531938000000000,
            0.120210000000000,
            0.122277000000000,
            0.454859000000000,
            0.123878000000000,
            0.151254000000000,
            0.0947060000000000,
            0,
            0,
            0,
            0.103411000000000,
            0.175508000000000,
            0.180817000000000,
            0,
            0,
            0,
            0,
            0,
            0,
            0.114969000000000,
            0.175506000000000,
            0.160961000000000,
            0,
            0,
            0,
            0,
            0,
            0,
            0.123887000000000,
            0.0914854000000000,
            0.353121000000000,
            0.128600000000000,
            0.0924281000000000,
            0.161045000000000,
            0.116530000000000,
            0.0886916000000000,
            0.344055000000000,
            0,
            0,
            0]
    print(pose2bb(pose, 0.005, 1920, 1080))