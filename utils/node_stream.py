import numpy as np
from utils.fifo import Fifo


class NodeStream:
    """
    The data streams of all the nodes in the system. Used by the DataGenerator class.
    Receives data points and updates the local vector of the nodes.
    Updates the global vector of the verifier with every new data point.
    Inherent class must implement update_local_vector method according to the local data aggregation method.
    """
    def __init__(self, num_nodes, sliding_window_size, data_point_len):
        self.sliding_window_size = sliding_window_size
        self.num_nodes = num_nodes
        self.sliding_windows = []
        self.local_vectors = []
        for node_idx in range(num_nodes):
            self.sliding_windows.append(Fifo(sliding_window_size, data_point_len))
            self.local_vectors.append(self.initial_x0.copy())
        self.global_vector = self.initial_x0.copy()

    def get_global_vector(self):
        return self.global_vector

    def get_local_vector(self, node_idx):
        return self.local_vectors[node_idx]

    def set_new_data_point(self, data_point, node_idx):
        self.local_vectors[node_idx] = self.update_local_vector(data_point, self.sliding_windows[node_idx], self.local_vectors[node_idx])
        self.global_vector = np.mean(self.local_vectors, axis=0)

    def all_windows_full(self):
        return np.alltrue([window.is_window_full() for window in self.sliding_windows])

    def node_window_full(self, node_idx):
        return self.sliding_windows[node_idx].is_window_full()

    def update_local_vector(self, data_point, sliding_window, x):
        raise NotImplementedError("To be implemented by inherent class")


class NodeStreamAverage(NodeStream):

    def __init__(self, num_nodes, sliding_window_size, data_point_len, x0_len):
        assert x0_len == data_point_len
        self.initial_x0 = np.zeros(x0_len)
        super().__init__(num_nodes, sliding_window_size, data_point_len)

    def update_local_vector(self, data_point, sliding_window, x):
        new_x = x
        num_samples = sliding_window.get_num_element()
        if sliding_window.is_window_full():
            # Remove the oldest sample
            oldest_data_point = sliding_window.get_oldest_element()
            new_x = (new_x * num_samples - oldest_data_point) / (num_samples - 1)
            num_samples -= 1

        # Add the new sample and increase num_samples
        sliding_window.add_element(data_point)
        num_samples += 1
        new_x = (new_x * (num_samples - 1) + data_point) / num_samples
        return new_x


class NodeStreamFrequency(NodeStream):

    def __init__(self, num_nodes, sliding_window_size, data_point_len, x0_len):
        assert data_point_len == 1
        self.initial_x0 = np.ones(x0_len, dtype=np.float) / x0_len
        super().__init__(num_nodes, sliding_window_size, data_point_len)

    def update_local_vector(self, data_point, sliding_window, x):
        # data_point is an integer, one of the k possible values (0 to k-1)
        new_x = x
        num_samples = sliding_window.get_num_element()
        new_x = np.around(new_x * num_samples)

        if sliding_window.is_window_full():
            # Remove the oldest sample
            oldest_data_point = sliding_window.get_oldest_element()
            new_x[int(oldest_data_point)] -= 1
            assert (new_x[int(oldest_data_point)] >= 0)

        # Add the new sample
        sliding_window.add_element(data_point)
        new_x[int(data_point)] += 1

        new_x = new_x / sliding_window.get_num_element()
        return new_x


class NodeStreamFirstAndSecondMomentum(NodeStream):

    def __init__(self, num_nodes, sliding_window_size, data_point_len, x0_len):
        assert data_point_len == 1
        assert x0_len == 2
        self.initial_x0 = np.zeros(x0_len)
        super().__init__(num_nodes, sliding_window_size, data_point_len)

    def update_local_vector(self, data_point, sliding_window, x):
        # data_point is a real value (scalar)
        new_x = x
        num_samples = sliding_window.get_num_element()
        if sliding_window.is_window_full():
            # Remove the oldest sample
            oldest_data_point = sliding_window.get_oldest_element()
            new_x[0] = (new_x[0] * num_samples - oldest_data_point) / (num_samples - 1)
            new_x[1] = (new_x[1] * num_samples - oldest_data_point ** 2) / (num_samples - 1)
            num_samples -= 1

        # Add the new sample and increase num_samples
        sliding_window.add_element(data_point)
        num_samples += 1
        new_x[0] = (new_x[0] * (num_samples - 1) + data_point) / num_samples
        new_x[1] = (new_x[1] * (num_samples - 1) + data_point ** 2) / num_samples
        return new_x


class NodeStreamConcatenatedFrequencyVectors(NodeStream):

    def __init__(self, num_nodes, sliding_window_size, data_point_len, x0_len):
        assert data_point_len == 2
        k = x0_len // 2  # Two concatenated frequency vectors
        self.initial_x0 = np.ones(x0_len, dtype=np.float) / k
        super().__init__(num_nodes, sliding_window_size, data_point_len)

    def update_local_vector(self, data_point, sliding_window, x):
        # data_point is a concatenation of two integers x1 and x2, where x1 and x2 are one of the k possible values (0 to k-1)
        new_x = x
        num_samples = sliding_window.get_num_element()
        new_x = np.around(new_x * num_samples)
        k = new_x.shape[0] // 2

        if sliding_window.is_window_full():
            # Remove the oldest sample
            oldest_data_point = sliding_window.get_oldest_element()
            new_x[int(oldest_data_point[0])] -= 1
            new_x[int(oldest_data_point[1]) + k] -= 1
            assert (new_x[int(oldest_data_point[0])] >= 0)
            assert (new_x[int(oldest_data_point[1]) + k] >= 0)

        # Add the new sample
        sliding_window.add_element(data_point)
        new_x[int(data_point[0])] += 1
        new_x[int(data_point[1]) + k] += 1

        new_x = new_x / sliding_window.get_num_element()
        return new_x
