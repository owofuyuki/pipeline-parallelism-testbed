import numpy as np
import networkx as nx
import utils

## DenseNet169
size = np.array([1572864, 33554432, 134217728, 134217728, 16777216, 67108864, 67108864, 8388608, 41943040, 41943040,
                 5242880, 13631488, 13631488, 851968, 851968, 5120])
exec_time = np.array([0.00000000e+00, 6.64709296e-03, 5.37520017e-01, 2.43449892e-02,
                      4.93398224e-02, 3.98106098e-01, 1.24339376e-02, 2.77071851e-02,
                      5.06166748e-01, 7.45320320e-03, 2.76232617e-02, 1.82106171e-01,
                      2.97844410e-03, 1.76719257e-03, 2.83888408e-05, 1.09587397e-04])

### ResNet50
size = np.array([1572864, 8388608, 8388608, 8388608,
                2097152, 8388608, 8388608, 4194304,
                4194304, 2097152, 2097152, 1048576,
                1048576, 1048576, 1048576, 5120])
exec_time = np.array([0.00000000e+00, 2.06923485e-03, 1.22609138e-03, 8.65364075e-04,
                     1.69405937e-03, 1.04828358e-02, 1.74146493e-02, 1.01133029e-02,
                     1.29773299e-02, 8.33187103e-03, 2.23259131e-02, 1.36006196e-02,
                     1.73825264e-02, 1.21577581e-04, 1.97569529e-05, 8.63393148e-05])

### ResNet101



if __name__ == "__main__":
    # change s to number of split DNN
    S_L = utils.all_partition_case(s=1, num_layer=len(size))
    S_total = []
    for cut_layer in S_L:
        G = nx.Graph()

        # initial device with speed argument
        G.add_nodes_from([('A', {'speed': 2.75408874427}), ('B', {'speed': 1})])

        # initial device graph with layer properties
        G.nodes['A']['start'] = 0
        G.nodes['A']['end'] = cut_layer[0]
        G.nodes['A']['e'] = 1
        G.nodes['B']['start'] = cut_layer[0]
        G.nodes['B']['end'] = -1
        G.nodes['B']['e'] = 1

        # initial edge with speed (BW in MBps)
        G.add_edge('A', 'B', speed=3375000*3)

        utils.process_algorithm(G, size, exec_time)

        S_total.append(utils.get_min_time(G))

    S_total = np.array(S_total)
    print(S_total)
    print(f"The optimal partition: L = {S_L[np.argmin(S_total)]}, total time = {np.min(S_total)}")
