import numpy as np
import networkx as nx


def get_min_time(G):
    t_total = []
    for d in G.nodes:
        t_total.append(G.nodes[d]['exec'] + G.nodes[d]['trans'])

    return np.max(t_total)


def process_algorithm(G, size, exec_time):
    for d in G.nodes:
        if G.nodes[d]['end'] != -1:
            G.nodes[d]['exec'] = np.sum(exec_time[G.nodes[d]['start']:G.nodes[d]['end'] + 1]) * G.nodes[d]['speed'] /\
                                 G.nodes[d]['e']
        else:
            G.nodes[d]['exec'] = np.sum(exec_time[G.nodes[d]['start']:]) * G.nodes[d]['speed'] / G.nodes[d]['e']

        G.nodes[d]['trans'] = 0.0
        for n in G.neighbors(d):
            layer = None
            if G.nodes[d]['start'] == G.nodes[n]['end']:
                layer = G.nodes[d]['start']
            if G.nodes[d]['end'] == G.nodes[n]['start']:
                layer = G.nodes[d]['end']
            G.nodes[d]['trans'] += size[layer] / (G[d][n]['speed'] * G.nodes[n]['e'])

        # dev[d]['data'] = size[]


def all_partition_case(s, num_layer):
    data = []
    if s == 1:
        for i in range(1, num_layer):
            data.append([i])
    if s == 2:
        for i in range(1, num_layer):
            for j in range(i + 1, num_layer):
                data.append([i, j])
    if s == 3:
        for i in range(1, num_layer):
            for j in range(i + 1, num_layer):
                for k in range(j + 1, num_layer):
                    data.append([i, j, k])
    if s == 4:
        for i in range(1, num_layer):
            for j in range(i + 1, num_layer):
                for k in range(j + 1, num_layer):
                    for l in range(k + 1, num_layer):
                        data.append([i, j, k, l])

    return data