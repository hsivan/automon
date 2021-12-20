import numpy as np
import enum
import os
from datasets.air_quality.read_csv import prepare_pm_data
from datasets.intrusion_detection.read_csv import prepare_intrusion_detection_data
from utils.node_stream import NodeStreamAverage, NodeStreamFirstAndSecondMomentum, NodeStreamFrequency, NodeStreamConcatenatedFrequencyVectors


class DataGeneratorState(enum.Enum):
    Monitoring = 0
    NeighborhoodTuning = 1


class DataGenerator:
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=1, test_folder="./", num_iterations_for_tuning=0, NodeStreamClass=None, sliding_window_size=100):
        self.num_iterations = num_iterations
        self.num_nodes = num_nodes
        self.d = d
        self.num_iterations_for_tuning = num_iterations_for_tuning
        self.NodeStreamClass = NodeStreamClass
        self.sliding_window_size = sliding_window_size
        self.state = DataGeneratorState.Monitoring
        self.node_schedule_file_name = test_folder + "/node_schedule_file.txt"

        if data_file_name is not None:
            self.data_file_name = test_folder + "/" + data_file_name
            self._get_data_from_file()
        else:
            self.data_file_name = test_folder + "/data_file.txt"
            self._generate_data_tuning_and_monitoring()
            self._save_data_to_file()

        self.reset()

    # Reset the read ptr
    def reset(self):
        self.data_ptr_tuning = 0
        self.data_ptr_monitoring = 0
        # Create data stream for each node. Each data stream sample is the activation of local_vector_update function on the samples in the sliding window of the specific node.
        # Before the first sample is retrieved from the stream, the sliding window must be full.
        self.node_stream = self.NodeStreamClass(self.num_nodes, self.sliding_window_size, self.get_data_point_len(), self.d)
        self.fill_all_sliding_windows()

    def _get_data_from_file(self):
        self.data = np.genfromtxt(self.data_file_name)
        self.tuning_data = self.data[:self.num_iterations_for_tuning*self.num_nodes]
        self.monitoring_data = self.data[self.num_iterations_for_tuning*self.num_nodes:]

        # write and read the schedule to file to support different types of scheduling.
        # Currently the only non-standard schedule is for DNN Intrusion Detection experiment.
        if os.path.exists(self.node_schedule_file_name):
            self.node_schedule = np.genfromtxt(self.node_schedule_file_name)
            self.node_schedule = self.node_schedule.astype(int)
            self.tuning_node_schedule = self.node_schedule[:self.num_iterations_for_tuning*self.num_nodes]
            self.monitoring_node_schedule = self.node_schedule[self.num_iterations_for_tuning*self.num_nodes:]
        else:
            self.tuning_node_schedule = np.tile(np.arange(self.num_nodes), self.num_iterations_for_tuning)
            self.monitoring_node_schedule = np.tile(np.arange(self.num_nodes), self.num_iterations)
            self.node_schedule = np.concatenate((self.tuning_node_schedule, self.monitoring_node_schedule))

    def _save_data_to_file(self):
        np.savetxt(self.data_file_name, self.data)
        np.savetxt(self.node_schedule_file_name, self.node_schedule)

    def _generate_data_tuning_and_monitoring(self):
        self.tuning_node_schedule = np.tile(np.arange(self.num_nodes), self.num_iterations_for_tuning)
        self.monitoring_node_schedule = np.tile(np.arange(self.num_nodes), self.num_iterations)

        self.tuning_data = self._generate_data(number_of_data_point=self.num_iterations_for_tuning*self.num_nodes, data_type="tuning")
        self.monitoring_data = self._generate_data(number_of_data_point=self.num_iterations*self.num_nodes, data_type="testing")

        self.data = np.concatenate((self.tuning_data, self.monitoring_data))
        # _generate_data() function may change the node schedule.
        # Therefore concatenation of tuning_node_schedule and monitoring_node_schedule must be after the call to _generate_data().
        self.node_schedule = np.concatenate((self.tuning_node_schedule, self.monitoring_node_schedule))

    def _generate_data(self, number_of_data_point, data_type):
        raise NotImplementedError("To be implemented by inherent class")

    def _get_next_data_point(self):
        if self.state == DataGeneratorState.NeighborhoodTuning:
            data_point = self.tuning_data[self.data_ptr_tuning]
            node_idx = self.tuning_node_schedule[self.data_ptr_tuning]
            self.data_ptr_tuning += 1
        else:
            data_point = self.monitoring_data[self.data_ptr_monitoring]
            node_idx = self.monitoring_node_schedule[self.data_ptr_monitoring]
            self.data_ptr_monitoring += 1
        return data_point, node_idx

    def get_next_data_point(self):
        data_point, node_idx = self._get_next_data_point()
        self.node_stream.set_new_data_point(data_point, node_idx)
        local_vector = self.node_stream.get_local_vector(node_idx)
        return local_vector, node_idx

    def get_global_vector(self):
        return self.node_stream.get_global_vector()

    def get_local_vector(self, node_idx):
        return self.node_stream.get_local_vector(node_idx)

    def get_num_samples(self):
        if self.state == DataGeneratorState.NeighborhoodTuning:
            return self.tuning_data.shape[0]
        else:
            return self.monitoring_data.shape[0]

    def get_num_iterations(self):
        if self.state == DataGeneratorState.NeighborhoodTuning:
            return self.num_iterations_for_tuning
        else:
            return self.num_iterations

    def set_neighborhood_tuning_state(self):
        self.state = DataGeneratorState.NeighborhoodTuning
        self.reset()

    def set_monitoring_state(self):
        self.state = DataGeneratorState.Monitoring
        self.reset()

    def get_data_point_len(self):
        # The difference between self.d and data_point_len is that d is the dimension of the local vector of a node, while
        # data_point_len is the dimension of the data sample the nodes receives. For example, in Entropy d is k (number of
        # classes/buckets) and data_point_len is 1 (data sample is a class index).
        data_point_len = 1 if len(self.data.shape) == 1 else self.data.shape[1]
        return data_point_len

    def has_next(self):
        if self.state == DataGeneratorState.NeighborhoodTuning:
            b_has_next = self.data_ptr_tuning < self.tuning_data.shape[0]
        else:
            b_has_next = self.data_ptr_monitoring < self.monitoring_data.shape[0]
        return b_has_next

    def fill_all_sliding_windows(self):
        # Fill all sliding windows
        while not self.node_stream.all_windows_full():
            data_point, idx = self._get_next_data_point()
            self.node_stream.set_new_data_point(data_point, int(idx))


class DataGeneratorVariance(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        # In variance d here is the dimension (length of the nodes' local vector), which is 2 - first and second moments.
        # However, the length of the data vector is 1.
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, 2, test_folder, num_iterations_for_tuning, NodeStreamFirstAndSecondMomentum, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data = np.random.randn(number_of_data_point)
        return data


class DataGeneratorEntropy(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=2, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamFrequency, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        a = np.arange(self.d)
        p = np.random.randint(0, high=10, size=self.d)
        assert (np.sum(p) > 0)
        data = np.zeros(number_of_data_point)
        p_high = 0.3
        batch_size = 50
        for i in range(0, number_of_data_point, batch_size):
            p = np.zeros(self.d)
            p_high = (p_high + 0.05) % 1
            p_low = (1 - p_high) / (self.d - 1)
            p += p_low
            p[0] = p_high
            num_points_in_batch = np.min((batch_size, number_of_data_point - i))
            data[i:i + num_points_in_batch] = np.random.choice(a, num_points_in_batch, replace=True, p=p)
        return data


class DataGeneratorInnerProduct(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=2, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        assert d % 2 == 0
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamAverage, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data = np.zeros((number_of_data_point, self.d))
        if data_type != "testing" and self.num_iterations_for_tuning == 0:
            return data

        # 5 parts:
        # 1 - straight diagonal line from 0 to 1
        # 2 - sine wave with large frequency
        # 3 - straight horizontal line
        # 4 - sine wave with small frequency
        # 5 - straight diagonal line from f to f+1
        number_of_data_point_per_part = number_of_data_point // 5
        f_res_part_1 = np.linspace(0, 1, number_of_data_point_per_part)
        f_res_part_2 = np.sin(np.linspace(0, 10, number_of_data_point_per_part)) + f_res_part_1[-1]
        f_res_part_3 = np.linspace(f_res_part_2[-1], f_res_part_2[-1], number_of_data_point_per_part)
        f_res_part_3[::2] += 1
        f_res_part_4 = np.sin(np.linspace(0, 40, number_of_data_point_per_part)) + f_res_part_3[-1]
        f_res_part_5 = np.linspace(f_res_part_4[-1], f_res_part_4[-1]+1, number_of_data_point - 4*number_of_data_point_per_part)
        f_res = np.concatenate((f_res_part_1, f_res_part_2, f_res_part_3, f_res_part_4, f_res_part_5))

        x = np.random.normal(loc=1, scale=0.01, size=(number_of_data_point, self.d // 2))

        y = np.zeros_like(x)
        for i in range(number_of_data_point):
            # Get loc such that <x[i], y[i]> = f_res[i]. We assume all y[i]_j are similar.
            loc = f_res[i] / np.sum(x[i])
            y[i] = np.random.normal(loc=loc, scale=0.01, size=(1, self.d // 2))
        data = np.append(x, y, axis=1)
        return data


class DataGeneratorCosineSimilarity(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=2, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        assert d % 2 == 0
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamAverage, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data_x = np.random.randn(number_of_data_point, self.d // 2) + 2
        data_y = np.random.randn(number_of_data_point, self.d // 2) + 4
        data = np.concatenate((data_x, data_y), axis=1)
        return data


class DataGeneratorQuadratic(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=1, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamAverage, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data = np.zeros((number_of_data_point, self.d))
        batch_size = 40
        b_odd_batch = False
        for node_idx in range(self.num_nodes):
            for i in range(0, number_of_data_point, batch_size * self.num_nodes):
                num_points_in_batch = min(batch_size, (number_of_data_point - (i + node_idx)) // self.num_nodes)
                data[i + node_idx:i + node_idx + self.num_nodes * num_points_in_batch:self.num_nodes, :] = np.random.normal(loc=0, scale=0.1, size=(num_points_in_batch, self.d))

                if node_idx == 0:
                    if b_odd_batch:
                        data[i + node_idx, :] = np.random.normal(loc=-10, scale=0.1, size=(1, self.d))
                    b_odd_batch = not b_odd_batch
        return data


class DataGeneratorRozenbrock(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=2, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        assert d % 2 == 0
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamAverage, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data_x = np.random.normal(loc=0.0, scale=0.2, size=(number_of_data_point, self.d // 2))
        data_y = np.random.normal(loc=0.0, scale=0.2, size=(number_of_data_point, self.d // 2))
        data = np.concatenate((data_x, data_y), axis=1)
        return data


class DataGeneratorKldAirQuality(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=52, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        assert d % 2 == 0
        if data_file_name is None:
            # Binning the data into d//2 bins. Q has d//2 bins and P has d//2 bins, and overall the dimension is d.
            self.station_data_arr = prepare_pm_data(step=500/((d // 2) - 1))
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamConcatenatedFrequencyVectors, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        num_stations = len(self.station_data_arr)
        data = np.zeros((number_of_data_point, 2))

        if data_type == "testing":
            stations_data = np.zeros((num_stations, self.num_iterations, 2))
        else:  # Tuning
            stations_data = np.zeros((num_stations, self.num_iterations_for_tuning, 2))

        for idx, station_data in enumerate(self.station_data_arr):
            if data_type == "testing":
                stations_data[idx, :, :] = station_data[self.num_iterations_for_tuning:self.num_iterations_for_tuning + self.num_iterations, :]
            else:  # Tuning
                stations_data[idx, :, :] = station_data[:self.num_iterations_for_tuning, :]

        data_counter = 0
        num_iterations = number_of_data_point // self.num_nodes
        for i in range(num_iterations):
            for station in range(num_stations):
                data[data_counter, :] = stations_data[station, i, :]
                data_counter += 1
        return data


class DataGeneratorMlp(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=5, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamAverage, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        # The global maximum of the function is at (1/sqrt(2), 0), and the global minimum (-1/sqrt(2), 0).
        # Half of the nodes have y=2, and the other half have y=-2.
        # The x is the same for all the nodes and it moves from -2 toward 2 a little bit every batch.
        # Every node stays on almost the same contour line while moving.
        data = np.zeros((number_of_data_point, self.d))
        batch_size_per_node = 2
        batch_size = batch_size_per_node * self.num_nodes
        move_per_batch = 0.008

        x_loc = -2
        y_loc_even = 2
        y_loc_odd = -2

        for i in range(0, 20 * self.num_nodes):
            loc = (x_loc,) + (0,) * (self.d - 1)
            data[i] = np.random.normal(loc=loc, scale=0.1, size=(1, self.d))

        for i in range(0, number_of_data_point - 20 * self.num_nodes):
            node_idx = i % self.num_nodes
            batch_idx = i // batch_size
            if i > 0 and i % batch_size == 0:  # Move to the next batch
                x_loc = x_loc + move_per_batch

            if node_idx % 2 == 0:
                loc = (x_loc,) + (y_loc_even,) * (self.d - 1)
            else:
                loc = (x_loc,) + (y_loc_odd,) * (self.d - 1)
            data[i + 20 * self.num_nodes] = np.random.normal(loc=loc, scale=0.1, size=(1, self.d))

            # Some rapid change for a short while to cause periodic oracle have small period
            if (360 < batch_idx < 370) or (380 < batch_idx < 390):
                if node_idx % 2 == 0:
                    loc = (0,) + (y_loc_even,) * (self.d - 1)
                    data[i + 20 * self.num_nodes] = np.random.normal(loc=loc, scale=0.05, size=(1, self.d))
                else:
                    loc = (0,) + (y_loc_odd,) * (self.d - 1)
                    data[i + 20 * self.num_nodes] = np.random.normal(loc=loc, scale=0.05, size=(1, self.d))
            if 390 < batch_idx < 400 and node_idx % 2 == 0:
                loc = (x_loc,) + (y_loc_even - 1.5,) * (self.d - 1)
                data[i + 20 * self.num_nodes] = np.random.normal(loc=loc, scale=0.1, size=(1, self.d))

        return data


class DataGeneratorSine(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=1, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamAverage, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data = np.zeros((number_of_data_point, self.d))
        loc = np.linspace(0, 8*np.pi, number_of_data_point)

        for i in range(0, number_of_data_point):
            data[i] = np.random.normal(loc=loc[i], scale=0.2)
        return data


class DataGeneratorDnnIntrusionDetection(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, d=41, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamAverage, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data = np.zeros((number_of_data_point, self.d + 1))  # Index 0 in every sample is the node idx that this sample goes to

        if data_type != "testing" and self.num_iterations_for_tuning == 0:
            return data

        if data_type == "testing":
            data = prepare_intrusion_detection_data(num_rows_to_drop_from_beginning=self.num_iterations_for_tuning * self.num_nodes)[:self.num_iterations]
            # Override the default node schedule
            self.monitoring_node_schedule = data[:, 0]
            data = data[:, 1:]
        else:  # Tuning
            data = prepare_intrusion_detection_data(num_rows_to_drop_from_beginning=0)[:number_of_data_point]
            # Override the default node schedule
            self.tuning_node_schedule = data[:, 0]
            data = data[:, 1:]

        return data


class DataGeneratorQuadraticInverse(DataGenerator):
    def __init__(self, num_iterations, num_nodes=4, data_file_name=None, d=2, test_folder="./", num_iterations_for_tuning=0, sliding_window_size=100):
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, d, test_folder, num_iterations_for_tuning, NodeStreamAverage, sliding_window_size)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        number_of_data_point_per_node = number_of_data_point // self.num_nodes
        end_point_inverse_trails = 1.6
        end_point_diagonal_trails = 1

        data = np.zeros((number_of_data_point, self.d))
        if data_type != "testing" and self.num_iterations_for_tuning == 0:
            return data

        # Node 0 goes from 0,0 to end_point_diagonal_trails,end_point_diagonal_trails
        loc_x = np.linspace(0, end_point_diagonal_trails, number_of_data_point_per_node)
        loc_y = np.linspace(0, end_point_diagonal_trails, number_of_data_point_per_node)
        node_0_data = np.zeros(shape=(number_of_data_point_per_node, 2))
        for i in range(20, number_of_data_point_per_node):
            node_0_data[i] = np.random.normal(loc=(loc_x[i], loc_y[i]), scale=0.01, size=(1, 2))

        # Node 1 goes from 0,0 to end_point_diagonal_trails,-end_point_diagonal_trails
        loc_x = np.linspace(0, end_point_diagonal_trails, number_of_data_point_per_node)
        loc_y = np.linspace(0, -end_point_diagonal_trails, number_of_data_point_per_node)
        node_1_data = np.zeros(shape=(number_of_data_point_per_node, 2))
        for i in range(20, number_of_data_point_per_node):
            node_1_data[i] = np.random.normal(loc=(loc_x[i], loc_y[i]), scale=0.01, size=(1, 2))

        # Node 2 goes from 0,0 to end_point_inverse_trails,0
        loc_x = np.linspace(0, end_point_inverse_trails, number_of_data_point_per_node)
        node_2_data = np.zeros(shape=(number_of_data_point_per_node, 2))
        for i in range(20, number_of_data_point_per_node):
            node_2_data[i] = np.random.normal(loc=(loc_x[i], 0), scale=(0, 0.01), size=(1, 2))

        # Node 3 goes from 0,0 to -end_point_inverse_trails,0
        loc_x = np.linspace(0, -end_point_inverse_trails, number_of_data_point_per_node)
        node_3_data = np.zeros(shape=(number_of_data_point_per_node, 2))
        for i in range(20, number_of_data_point_per_node):
            node_3_data[i] = np.random.normal(loc=(loc_x[i], 0), scale=(0, 0.01), size=(1, 2))

        # Some rapid change in the inverse trails
        start_change_point = (number_of_data_point_per_node // 3) * 2
        change_length = 10
        node_0_data[start_change_point:start_change_point + change_length] = np.random.normal(loc=(0.2, 0.2), scale=0.01, size=(change_length, 2))
        node_1_data[start_change_point:start_change_point + change_length] = np.random.normal(loc=(0.2, -0.2), scale=0.001, size=(change_length, 2))
        node_0_data[start_change_point+30:start_change_point+30 + change_length] = np.random.normal(loc=(0.3, 0.3), scale=0.01, size=(change_length, 2))
        node_1_data[start_change_point+30:start_change_point+30 + change_length] = np.random.normal(loc=(0.3, -0.3), scale=0.01, size=(change_length, 2))

        data[0::4] = node_0_data
        data[1::4] = node_1_data
        data[2::4] = node_2_data
        data[3::4] = node_3_data
        return data
