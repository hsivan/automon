from automon.fifo import Fifo
import numpy as np


class NodeStream:
    """
    The data streams of all the nodes in the system.
    Receives data points and updates the local vector of the nodes.
    Updates the global vector of the verifier with every new data point.
    """
    def __init__(self, num_nodes, sliding_window_size, data_point_len, func_update_local_vector, initial_x0):
        self.sliding_window_size = sliding_window_size
        self.num_nodes = num_nodes
        self.func_update_local_vector = func_update_local_vector
        self.sliding_windows = []
        self.local_vectors = []
        for node_idx in range(num_nodes):
            self.sliding_windows.append(Fifo(sliding_window_size, data_point_len))
            self.local_vectors.append(initial_x0.copy())
        self.global_vector = initial_x0.copy()

    def get_global_vector(self):
        return self.global_vector

    def get_local_vector(self, node_idx):
        return self.local_vectors[node_idx]

    def set_new_data_point(self, data_point, node_idx):
        self.local_vectors[node_idx] = self.func_update_local_vector(data_point, self.sliding_windows[node_idx], self.local_vectors[node_idx])
        self.global_vector = np.mean(self.local_vectors, axis=0)

    def all_windows_full(self):
        return np.alltrue([window.is_window_full() for window in self.sliding_windows])

    def node_window_full(self, node_idx):
        return self.sliding_windows[node_idx].is_window_full()
