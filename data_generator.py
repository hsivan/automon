import numpy as np
import os
from datasets.air_quality.read_data import prepare_pm_data
from datasets.intrusion_detection.read_csv import prepare_intrusion_detection_data


class DataGenerator:
    def __init__(self, num_iterations, num_nodes, data_file_name=None, test_folder="./", num_iterations_for_tuning=0):
        self.num_iterations = num_iterations
        self.num_nodes = num_nodes
        self.num_iterations_for_tuning = num_iterations_for_tuning

        if data_file_name is not None:
            self.data_file_name = test_folder + "/" + data_file_name
            self._get_data_from_file()
        else:
            self.data_file_name = test_folder + "/data_file.txt"
            self._generate_data_tune_and_test()

            self._save_data_to_file()

        self.data_ptr = 0
        return

    def _get_data_from_file(self):
        self.data = np.genfromtxt(self.data_file_name)

    def _save_data_to_file(self):
        np.savetxt(self.data_file_name, self.data)

    def _generate_data_tune_and_test(self):
        data_for_tuning = self._generate_data(number_of_data_point=self.num_iterations_for_tuning*self.num_nodes, data_type="tuning")
        data_for_test = self._generate_data(number_of_data_point=self.num_iterations*self.num_nodes, data_type="testing")
        self.data = np.concatenate((data_for_tuning, data_for_test))

    def _generate_data(self, number_of_data_point):
        raise NotImplementedError("To be implemented by inherent class")

    def get_next_data_point(self):
        data_point = self.data[self.data_ptr]
        self.data_ptr += 1
        return data_point

    # Reset the read ptr to point on the test data (skip tuning data)
    def reset(self):
        self.data_ptr = self.num_iterations_for_tuning * self.num_nodes

    def get_num_samples(self):
        if self.data_ptr == 0:  # Tuning data
            return self.num_iterations_for_tuning * self.num_nodes
        else:  # Testing data
            return self.data.shape[0] - self.num_iterations_for_tuning * self.num_nodes

    # Reset the read ptr to point on the tuning data
    def reset_for_tuning(self):
        self.data_ptr = 0


class DataGeneratorVariance(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, test_folder="./", num_iterations_for_tuning=0):
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data = np.random.randn(number_of_data_point)
        return data


class DataGeneratorEntropy(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, k=2, test_folder="./", num_iterations_for_tuning=0):
        self.k = k
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        a = np.arange(self.k)
        p = np.random.randint(0, high=10, size=self.k)
        assert (np.sum(p) > 0)
        data = np.zeros(number_of_data_point)
        p_high = 0.3
        batch_size = 50
        for i in range(0, number_of_data_point, batch_size):
            p = np.zeros(self.k)
            p_high = (p_high + 0.05) % 1
            p_low = (1 - p_high) / (self.k - 1)
            p += p_low
            p[0] = p_high
            num_points_in_batch = np.min((batch_size, number_of_data_point - i))
            data[i:i + num_points_in_batch] = np.random.choice(a, num_points_in_batch, replace=True, p=p)
        return data


class DataGeneratorInnerProduct(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, k=1, test_folder="./", num_iterations_for_tuning=0):
        self.k = k
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data = np.zeros((number_of_data_point, 2 * self.k))
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

        x = np.random.normal(loc=1, scale=0.01, size=(number_of_data_point, self.k))

        y = np.zeros_like(x)
        for i in range(number_of_data_point):
            # Get loc such that <x[i], y[i]> = f_res[i]. We assume all y[i]_j are similar.
            loc = f_res[i] / np.sum(x[i])
            y[i] = np.random.normal(loc=loc, scale=0.01, size=(1, self.k))
        data = np.append(x, y, axis=1)
        return data


class DataGeneratorQuadratic(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, k=1, test_folder="./", num_iterations_for_tuning=0):
        self.k = k
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data = np.zeros((number_of_data_point, self.k))
        batch_size = 40
        b_odd_batch = False
        for node_idx in range(self.num_nodes):
            for i in range(0, number_of_data_point, batch_size * self.num_nodes):
                num_points_in_batch = min(batch_size, (number_of_data_point - (i + node_idx)) // self.num_nodes)
                data[i + node_idx:i + node_idx + self.num_nodes * num_points_in_batch:self.num_nodes, :] = np.random.normal(loc=0, scale=0.1, size=(num_points_in_batch, self.k))

                if node_idx == 0:
                    if b_odd_batch:
                        data[i + node_idx, :] = np.random.normal(loc=-10, scale=0.1, size=(1, self.k))
                    b_odd_batch = not b_odd_batch
        return data


class DataGeneratorRozenbrock(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, k=1, test_folder="./", num_iterations_for_tuning=0):
        self.k = k
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data_x = np.random.normal(loc=0.0, scale=0.2, size=(number_of_data_point, self.k))
        data_y = np.random.normal(loc=0.0, scale=0.2, size=(number_of_data_point, self.k))
        data = np.concatenate((data_x, data_y), axis=1)
        return data


class DataGeneratorKldAirQuality(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, k=26, test_folder="./", num_iterations_for_tuning=0):
        self.k = k
        prepare_pm_data(step=500/(k-1), relative_folder='datasets/air_quality/')
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        dataset_folder = 'datasets/air_quality'
        data_files = [dataset_folder + "/" + file for file in os.listdir(dataset_folder) if file.startswith('station') and file.endswith("_pm.txt")]
        num_stations = len(data_files)
        data = np.zeros((number_of_data_point, 2))

        if data_type == "testing":
            stations_data = np.zeros((num_stations, self.num_iterations, 2))
        else:  # Tuning
            stations_data = np.zeros((num_stations, self.num_iterations_for_tuning, 2))

        for idx, station_file in enumerate(data_files):
            station_data = np.genfromtxt(station_file)
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
    def __init__(self, num_iterations, num_nodes, data_file_name=None, k=5, test_folder="./", num_iterations_for_tuning=0):
        self.k = k
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        # The global maximum of the function is at (1/sqrt(2), 0), and the global minimum (-1/sqrt(2), 0).
        # Half of the nodes have y=2, and the other half have y=-2.
        # The x is the same for all the nodes and it moves from -2 toward 2 a little bit every batch.
        # Every node stays on almost the same contour line while moving.
        data = np.zeros((number_of_data_point, self.k))
        batch_size_per_node = 2
        batch_size = batch_size_per_node * self.num_nodes
        move_per_batch = 0.008

        x_loc = -2
        y_loc_even = 2
        y_loc_odd = -2

        for i in range(0, number_of_data_point):
            node_idx = i % self.num_nodes
            batch_idx = i // batch_size
            if i > 0 and i % batch_size == 0:  # Move to the next batch
                x_loc = x_loc + move_per_batch

            if node_idx % 2 == 0:
                loc = (x_loc,) + (y_loc_even,) * (self.k - 1)
            else:
                loc = (x_loc,) + (y_loc_odd,) * (self.k - 1)
            data[i] = np.random.normal(loc=loc, scale=0.1, size=(1, self.k))

            # Some rapid change for a short while to cause periodic oracle have small period
            if (360 < batch_idx < 370) or (380 < batch_idx < 390):
                if node_idx % 2 == 0:
                    loc = (0,) + (y_loc_even,) * (self.k - 1)
                    data[i] = np.random.normal(loc=loc, scale=0.05, size=(1, self.k))
                else:
                    loc = (0,) + (y_loc_odd,) * (self.k - 1)
                    data[i] = np.random.normal(loc=loc, scale=0.05, size=(1, self.k))
            if 390 < batch_idx < 400 and node_idx % 2 == 0:
                loc = (x_loc,) + (y_loc_even - 1.5,) * (self.k - 1)
                data[i] = np.random.normal(loc=loc, scale=0.1, size=(1, self.k))

        return data


class DataGeneratorSine(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, k=1, test_folder="./", num_iterations_for_tuning=0):
        self.k = k
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        data = np.zeros((number_of_data_point, self.k))
        loc = np.linspace(0, 8*np.pi, number_of_data_point)

        for i in range(0, number_of_data_point):
            data[i] = np.random.normal(loc=loc[i], scale=0.2)
        return data


class DataGeneratorDnnIntrusionDetection(DataGenerator):
    def __init__(self, num_iterations, num_nodes, data_file_name=None, k=41, test_folder="./", num_iterations_for_tuning=0):
        self.k = k
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        dataset_folder = 'datasets/intrusion_detection/'
        data = np.zeros((number_of_data_point, self.k+1))  # Index 0 in every sample is the node idx that this sample goes to

        if data_type != "testing" and self.num_iterations_for_tuning == 0:
            return data

        if data_type == "testing":
            data = prepare_intrusion_detection_data(relative_folder=dataset_folder, num_rows_to_drop_from_beginning=self.num_iterations_for_tuning*self.num_nodes)[:self.num_iterations]
        else:  # Tuning
            data = prepare_intrusion_detection_data(relative_folder=dataset_folder, num_rows_to_drop_from_beginning=0)[:number_of_data_point]

        return data


class DataGeneratorQuadraticInverse(DataGenerator):
    def __init__(self, num_iterations, num_nodes=4, data_file_name=None, k=2, test_folder="./", num_iterations_for_tuning=0):
        self.k = k
        DataGenerator.__init__(self, num_iterations, num_nodes, data_file_name, test_folder, num_iterations_for_tuning)

    def _generate_data(self, number_of_data_point, data_type="testing"):
        number_of_data_point_per_node = number_of_data_point // self.num_nodes
        end_point_inverse_trails = 1.6
        end_point_diagonal_trails = 1

        data = np.zeros((number_of_data_point, self.k))
        if data_type != "testing" and self.num_iterations_for_tuning == 0:
            return data

        # Node 0 goes from 0,0 to end_point_diagonal_trails,end_point_diagonal_trails
        loc_x = np.linspace(0, end_point_diagonal_trails, number_of_data_point_per_node)
        loc_y = np.linspace(0, end_point_diagonal_trails, number_of_data_point_per_node)
        node_0_data = np.zeros(shape=(number_of_data_point_per_node, 2))
        for i in range(0, number_of_data_point_per_node):
            node_0_data[i] = np.random.normal(loc=(loc_x[i], loc_y[i]), scale=0.01, size=(1, 2))

        # Node 1 goes from 0,0 to end_point_diagonal_trails,-end_point_diagonal_trails
        loc_x = np.linspace(0, end_point_diagonal_trails, number_of_data_point_per_node)
        loc_y = np.linspace(0, -end_point_diagonal_trails, number_of_data_point_per_node)
        node_1_data = np.zeros(shape=(number_of_data_point_per_node, 2))
        for i in range(0, number_of_data_point_per_node):
            node_1_data[i] = np.random.normal(loc=(loc_x[i], loc_y[i]), scale=0.01, size=(1, 2))

        # Node 2 goes from 0,0 to end_point_inverse_trails,0
        loc_x = np.linspace(0, end_point_inverse_trails, number_of_data_point_per_node)
        node_2_data = np.zeros(shape=(number_of_data_point_per_node, 2))
        for i in range(0, number_of_data_point_per_node):
            node_2_data[i] = np.random.normal(loc=(loc_x[i], 0), scale=(0, 0.01), size=(1, 2))

        # Node 3 goes from 0,0 to -end_point_inverse_trails,0
        loc_x = np.linspace(0, -end_point_inverse_trails, number_of_data_point_per_node)
        node_3_data = np.zeros(shape=(number_of_data_point_per_node, 2))
        for i in range(0, number_of_data_point_per_node):
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
