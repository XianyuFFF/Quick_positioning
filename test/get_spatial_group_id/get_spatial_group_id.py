from config_default import configs
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster

def get_spatial_group_id(use_grouping, current_detection_IDX, detection_centers, params):
    # spatial_group_IDs = np.ones((len(current_detection_IDX), 1))

    if use_grouping is True:
        pairwise_distance = cdist(detection_centers, detection_centers)
        agglomeration = linkage(pairwise_distance)
        num_spatial_groups = round(params['cluster_coeff'] * len(current_detection_IDX) / params['window_width'])
        num_spatial_groups = max(num_spatial_groups, 0)

        while True:
            spatial_group_IDs = fcluster(agglomeration, criterion='maxclust', t=num_spatial_groups)
            uid = np.unique(spatial_group_IDs)
            freq = np.hstack((np.histogram(spatial_group_IDs.flatten()), uid))

            largest_group_size = len(freq)

            # The BIP solver might run out of memory for large graph
            if largest_group_size <= 150:
                return spatial_group_IDs
            num_spatial_groups += 1

if __name__ == '__main__':
    use_grouping = True
    params = configs['tracklets']
    detection_centers = pd.read_csv('detection_centers.txt',sep='\t',header=None)
    detection_centers = detection_centers.to_numpy()
    print(detection_centers)

    current_detection_idx = pd.read_csv('current_detection_idx.txt', sep='\t', header=None)
    current_detection_idx = current_detection_idx.to_numpy()

    print(get_spatial_group_id(use_grouping, current_detection_idx, detection_centers, params))