import numpy as np
from . import tools

# Joint index:
# {0: "WRIST"},
# {1: "THUMB_CMC"},
# {2: "THUMB_MCP"},
# {3: "THUMB_IP"},
# {4: "THUMB_TIP"},
# {5: "INDEX_FINGER_MCP"},
# {6: "INDEX_FINGER_PIP"},
# {7: "INDEX_FINGER_DIP"},
# {8: "INDEX_FINGER_TIP"},
# {9: "MIDDLE_FINGER_MCP"},
# {10: "MIDDLE_FINGER_PIP"},
# {11: "MIDDLE_FINGER_DIP"},
# {12: "MIDDLE_FINGER_TIP"},
# {13: "RING_FINGER_MCP"},
# {14: "RING_FINGER_PIP"},
# {15: "RING_FINGER_DIP"},
# {16: "RING_FINGER_TIP"},
# {17: "PINKY_MCP"},
# {18: "PINKY_PIP"},
# {19: "PINKY_DIP"},
# {20: "PINKY_TIP"},


# Edge format: (origin, neighbor)
num_node = 21
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),(0,17)]
        
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix
    For more information, please refer to the section 'Partition Strategies' in our paper.
    """

    def __init__(self, labeling_mode='uniform'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = tools.get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = tools.get_DLD_graph(num_node, self_link, neighbor)
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()
        return A


def main():
    mode = ['uniform', 'distance*', 'distance', 'spatial', 'DAD', 'DLD']
    np.set_printoptions(threshold=np.nan)
    for m in mode:
        print('=' * 10 + m + '=' * 10)
        print(Graph(m).get_adjacency_matrix())


if __name__ == '__main__':
    main()