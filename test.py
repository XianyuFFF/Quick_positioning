import numpy as np
import pandas as pd
import KL
import networkx as nx


data = pd.read_csv('t_data.txt',sep='\t', header=None)
# out = pd.read_csv('out.txt', sep='\t', header=None)
# data.columns = ['a', 'b', 'weight']

weighted_graph = data.to_numpy()
# out_data = out.to_numpy()

nEdges = np.size(weighted_graph, 0)

weighted_graph = weighted_graph.T.flatten()

nNodes = 50

# print(weighted_graph)
# print(weighted_graph.shape)

A = KL.MyMultiKL(nNodes, nEdges, weighted_graph.tolist())

nodes = list(range(nNodes))
edegs = zip(A[0:-1:2], A[1:-1:2])
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edegs)
conncomps = nx.connected_components(G)

new_result = np.zeros(nNodes)

for i, conn_set in enumerate(conncomps):
    for node_idx in conn_set:
        new_result[node_idx] = i + 1


print(list(conncomps))
print(new_result)


# print(out_data)
# print(np.sum(np.asarray(result) != out_data))