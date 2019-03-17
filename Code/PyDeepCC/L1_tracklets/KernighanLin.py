import numpy as np
from utils import sub2ind
from functools import reduce
import KL
import networkx as nx

def KernighanLin(correlation_matrix):
    if np.size(correlation_matrix, 0) == 1:
        return [1]
    upper_tri = np.triu(np.ones_like(correlation_matrix))
    source_node, target_node = np.nonzero(upper_tri)
    values = correlation_matrix[upper_tri != 0]
    infidx = np.nonzero(values == -np.inf)[0]

    #TODO edit
    rm_idx = source_node == target_node
    source_node[rm_idx] = np.array([])
    target_node[rm_idx] = np.array([])
    values[rm_idx] = np.array([])

    n_nodes = reduce((lambda x, y: x*y), np.diag(upper_tri))
    weighted_graph = np.hstack((source_node, target_node, values)).T

    # weighted graph to identity is ok
    A = KL.Multicut_KL(n_nodes, np.size(weighted_graph, 1), weighted_graph.flatten().tolist())

    edegs = zip(A[0:-1:2], A[1:-1:2])
    G = nx.Graph()
    G.add_node(list(range(n_nodes)))
    G.add_edges_from(edegs)
    conncomps = nx.connected_components(G)

    new_result = np.zeros(n_nodes)

    for i, conn_set in enumerate(conncomps):
        for node_idx in conn_set:
            new_result[node_idx] = i + 1

    return new_result

