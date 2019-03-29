import numpy as np
from utils import sub2ind
from functools import reduce
import KL
import networkx as nx

def KernighanLin(correlation_matrix):
    if np.size(correlation_matrix, 0) == 1:
        return np.array([1])
    upper_tri = np.triu(np.ones_like(correlation_matrix))
    source_node, target_node = np.nonzero(upper_tri)
    values = correlation_matrix[upper_tri != 0]
    infidx = np.nonzero(values == -np.inf)[0]

    #TODO edit
    rm_idx = np.where(source_node == target_node)[0]
    source_node = np.delete(source_node, rm_idx)
    target_node = np.delete(target_node, rm_idx)
    values = np.delete(values, rm_idx)

    n_nodes = np.size(np.diag(upper_tri))
    n_edges = np.size(source_node, 0)
    weighted_graph = np.hstack((source_node, target_node, values)).tolist()

    # weighted graph to identity is ok
    A = KL.MyMultiKL(n_nodes, n_edges, weighted_graph)

    edegs = zip(A[0:-1:2], A[1:-1:2])
    G = nx.Graph()
    G.add_nodes_from(list(range(n_nodes)))
    G.add_edges_from(edegs)
    conncomps = nx.connected_components(G)

    new_result = np.zeros(n_nodes)

    for i, conn_set in enumerate(conncomps):
        for node_idx in conn_set:
            new_result[node_idx] = i + 1

    return new_result

