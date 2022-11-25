# 生成p矩阵
import numpy as np
import igraph as ig

def _graph_to_adjmat(G):
    return np.array(G.get_adjacency().data)

def generate_tri(num_nodes, graph_type):#num_nodes是变量数
    if graph_type == 'ER':
        w_decay = 1.0
        i = 1
        neg = 0.5
        if num_nodes < 10:
            degree = 2
        else:
            degree = float(num_nodes/10)
        u_i = np.random.uniform(low=0, high=2.0, size=[num_nodes, num_nodes]) / (w_decay ** i)
        u_i[np.random.rand(num_nodes, num_nodes) < neg] *= -1

        # ER图的设置
        prob = degree / num_nodes
        b = (np.random.rand(num_nodes, num_nodes) < prob).astype(float)

        # 生成a矩阵
        a = (b != 0).astype(float) * u_i

    elif graph_type == 'SBM':

        p_in = 0.3
        p_out = 0.3 * p_in
        part1 = int(0.5 * num_nodes)
        part2 = num_nodes - part1

        print(num_nodes)
        print(part1)
        print(part2)

        G = ig.Graph.SBM(n=num_nodes, pref_matrix=[[p_in, p_out], [p_out, p_in]], block_sizes=[part1, part2])
        a = _graph_to_adjmat(G)

    return a

num_nodes = 5
graph_type = 'ER'
er = generate_tri(num_nodes, graph_type)
print(er)
