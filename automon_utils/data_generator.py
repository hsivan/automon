import numpy as np
import pandas as pd
import enum
import os
from sklearn.preprocessing import Normalizer
from automon_utils.node_stream import NodeStreamAverage, NodeStreamFirstAndSecondMomentum, NodeStreamFrequency, NodeStreamConcatenatedFrequencyVectors


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
            self.station_data_arr = DataGeneratorKldAirQuality.prepare_pm_data(step=500/((d // 2) - 1))
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

    @staticmethod
    def read_csv(csv_file):
        csv_file_full_path = os.path.abspath(os.path.dirname(__file__)) + "/../datasets/air_quality/" + csv_file
        df_station = pd.read_csv(csv_file_full_path)
        df_station = df_station.drop(columns=['No', 'year', 'month', 'day', 'hour', 'wd', 'station'])
        df_station = df_station.dropna()
        return df_station

    @staticmethod
    def prepare_pm_data(step=20.0):
        csv_files = [csv_file for csv_file in os.listdir(os.path.abspath(os.path.dirname(__file__)) + "/../datasets/air_quality/") if
                     csv_file.startswith("PRSA") and csv_file.endswith('csv')]
        if len(csv_files) == 0:
            print("No kld dataset files found")
            raise Exception
        station_data_arr = []

        for csv_file in csv_files:
            station_name = csv_file.split('_')[2]
            print("station_name:", station_name)
            df_station = DataGeneratorKldAirQuality.read_csv(csv_file)
            # x vector is the TEMP (0 to 25) ans y vector is the DEWP (dew point temperature, -25 to 25)

            number_of_data_point = df_station['PM10'].shape[0]
            data = np.zeros((number_of_data_point, 2), dtype='int')
            x = df_station['PM10'].values
            y = df_station['PM2.5'].values

            x = x.clip(0, 500)
            bins_temp = np.arange(0, 500, step)
            indices_temp = np.digitize(x, bins_temp) - 1
            x = indices_temp

            y = y.clip(0, 500)
            bins_dewp = np.arange(0, 500, step)
            indices_dewp = np.digitize(y, bins_dewp) - 1
            y = indices_dewp

            data[:, 0] = x
            data[:, 1] = y
            station_data_arr.append(data)

        return station_data_arr


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
            data = DataGeneratorDnnIntrusionDetection.prepare_intrusion_detection_data(num_rows_to_drop_from_beginning=self.num_iterations_for_tuning * self.num_nodes)[:self.num_iterations]
            # Override the default node schedule
            self.monitoring_node_schedule = data[:, 0]
            data = data[:, 1:]
        else:  # Tuning
            data = DataGeneratorDnnIntrusionDetection.prepare_intrusion_detection_data(num_rows_to_drop_from_beginning=0)[:number_of_data_point]
            # Override the default node schedule
            self.tuning_node_schedule = data[:, 0]
            data = data[:, 1:]

        return data

    @staticmethod
    def get_data(file):
        df = pd.read_csv(file, header=None)

        # Change categorical columns to numerical values. Use the following mapping to keep the same mapping as in https://github.com/rahulvigneswaran/Intrusion-Detection-Systems

        # df[1]: transport layer protocol
        mapping = {'tcp': 1, 'udp': 2, 'icmp': 3}
        df = df.replace({1: mapping})
        # df[2]: application
        mapping = {'http': 1, 'smtp': 19, 'finger': 51, 'domain_u': 57, 'auth': 65, 'telnet': 13, 'ftp': 50,
                   'eco_i': 55,
                   'ntp_u': 30, 'ecr_i': 54, 'other': 29, 'private': 24, 'pop_3': 26, 'ftp_data': 49, 'rje': 21,
                   'time': 10,
                   'mtp': 38, 'link': 40, 'remote_job': 22, 'gopher': 48, 'ssh': 17, 'name': 37, 'whois': 4,
                   'domain': 58,
                   'login': 39, 'imap4': 46, 'daytime': 70, 'ctf': 61, 'nntp': 31, 'shell': 20, 'IRC': 45, 'nnsp': 32,
                   'http_443': 47, 'exec': 52, 'printer': 25, 'efs': 53, 'courier': 63, 'uucp': 7, 'klogin': 43,
                   'kshell': 42,
                   'echo': 56, 'discard': 59, 'systat': 14, 'supdup': 15, 'iso_tsap': 44, 'hostnames': 69,
                   'csnet_ns': 62,
                   'pop_2': 27, 'sunrpc': 16, 'uucp_path': 6, 'netbios_ns': 35, 'netbios_ssn': 34, 'netbios_dgm': 36,
                   'sql_net': 18,
                   'vmnet': 5, 'bgp': 64, 'Z39_50': 2, 'ldap': 41, 'netstat': 33, 'urh_i': 9, 'X11': 3, 'urp_i': 8,
                   'pm_dump': 28, 'tftp_u': 12, 'tim_i': 11, 'red_i': 23, 'icmp': 68}
        df = df.replace({2: mapping})
        # df[3]: connection flag
        mapping = {'SF': 1, 'SH': 2, 'S3': 3, 'OTH': 4, 'S2': 4, 'S1': 5, 'S0': 6, 'RSTR': 7, 'RSTOS0': 8, 'RSTO': 9,
                   'REJ': 10}
        df = df.replace({3: mapping})

        # Set label 'normal' to 0 and other labels to 1
        values = pd.unique(df[41])
        for value in values:
            if value == 'normal.':
                df[41].replace(value, 0, inplace=True)
            else:
                df[41].replace(value, 1, inplace=True)
        # Move last column to be the first
        cols = list(df.columns)
        cols = [cols[-1]] + cols[:-1]
        df = df[cols]
        col_names = dict([(i, i + 1) for i in range(41)])
        col_names[41] = 0
        df.rename(columns=col_names, inplace=True)
        return df

    @staticmethod
    # Data taken from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html kddcup.data_10_percent.gz
    def get_training_data():
        file_path = os.path.abspath(os.path.dirname(__file__)) + "/../datasets/intrusion_detection/kddcup.data_10_percent_corrected"
        return DataGeneratorDnnIntrusionDetection.get_data(file_path)

    @staticmethod
    # Data taken from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html corrected.gz
    def get_testing_data():
        data_stream = os.path.abspath(os.path.dirname(__file__)) + "/../datasets/intrusion_detection/corrected"
        return DataGeneratorDnnIntrusionDetection.get_data(data_stream)

    @staticmethod
    def set_nodes_data(df_, app_idx, node_indices, nodes):
        apps, counts = np.unique(df_[3], return_counts=True)
        app_type = apps[app_idx]
        num_nodes = len(nodes)

        app_type_data = df_[df_[3].isin(app_type)]
        for i, node in enumerate(nodes):
            node_data = app_type_data[i::num_nodes]  # Round-robin between the nodes that get this application rows
            node_indices[node_data.index] = nodes[i]
        return node_indices

    @staticmethod
    def prepare_intrusion_detection_data(num_rows_to_drop_from_beginning=0):
        sliding_window_size = 20
        num_nodes = 9

        df = DataGeneratorDnnIntrusionDetection.get_testing_data()
        apps, counts = np.unique(df[3], return_counts=True)
        # Divide 164352 rows with application index 51 to 5 nodes (32870 rows per node).
        # Divide 78510 rows with application index 21 to 2 nodes (39255 rows per node).
        # Put 41237 rows with application index 0 on single node.
        # Put the rest 26930 rows on single node.
        # Overall 9 nodes with at least 30000 samples each.
        counts[51]  # 164352 (Private)
        counts[21]  # 78510 (ECR_i)
        counts[0]  # 41237 (http)

        node_indices = np.zeros(df.shape[0])

        train_data = DataGeneratorDnnIntrusionDetection.get_training_data()
        train_x = train_data.iloc[:, 1:42].values
        scaler = Normalizer().fit(train_x)

        node_indices = DataGeneratorDnnIntrusionDetection.set_nodes_data(df, [51], node_indices, [0, 1, 2, 3, 4])
        node_indices = DataGeneratorDnnIntrusionDetection.set_nodes_data(df, [21], node_indices, [5, 6])
        node_indices = DataGeneratorDnnIntrusionDetection.set_nodes_data(df, [0], node_indices, [7])
        rest_app_indices = [i for i in range(len(apps)) if i != 51 and i != 21 and i != 0]
        node_indices = DataGeneratorDnnIntrusionDetection.set_nodes_data(df, rest_app_indices, node_indices, [8])[num_rows_to_drop_from_beginning:]

        nodes_data = df.values[num_rows_to_drop_from_beginning:][:, 1:42]
        nodes_data = scaler.transform(nodes_data)
        nodes_data = np.append(np.expand_dims(node_indices, -1), nodes_data, axis=1)
        max_index_of_full_window = 0
        for node_idx in range(num_nodes):
            index_of_full_window = np.argwhere(nodes_data[:, 0] == node_idx)[sliding_window_size][0]
            if index_of_full_window > max_index_of_full_window:
                max_index_of_full_window = index_of_full_window
        # For every node leave the last sliding_window_size samples before max_index_of_full_window in nodes_data
        num_rows_to_remove_per_node = []
        for node_idx in range(num_nodes):
            num_rows_to_remove = sum(np.argwhere(nodes_data[:, 0] == node_idx) < max_index_of_full_window)[
                                     0] - sliding_window_size
            num_rows_to_remove_per_node.append(num_rows_to_remove)
        all_rows_to_remove = []
        for node_idx in range(num_nodes):
            rows_to_remove = np.argwhere(nodes_data[:, 0] == node_idx).squeeze()[:num_rows_to_remove_per_node[node_idx]]
            all_rows_to_remove += rows_to_remove.tolist()
        nodes_data = np.delete(nodes_data, all_rows_to_remove, axis=0)
        # Take sliding_window_size*num_nodes first rows and make them interleaving: 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, ...
        initial_data = nodes_data[0:sliding_window_size * num_nodes].copy()
        for node_idx in range(num_nodes):
            initial_data[node_idx:sliding_window_size * num_nodes:num_nodes] = nodes_data[
                np.argwhere(nodes_data[0:sliding_window_size * num_nodes, 0] == node_idx).squeeze()]
        nodes_data[0:sliding_window_size * num_nodes] = initial_data

        return nodes_data


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
