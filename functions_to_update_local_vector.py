import numpy as np


def update_local_vector_average(data_point, sliding_window, x):
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


def update_local_vector_frequency(data_point, sliding_window, x):
    # data_point is an integer, one of the k possible values (0 to k-1)
    new_x = x
    num_samples = sliding_window.get_num_element()
    new_x = np.around(new_x * num_samples)

    if sliding_window.is_window_full():
        # Remove the oldest sample
        oldest_data_point = sliding_window.get_oldest_element()
        new_x[int(oldest_data_point)] -= 1
        assert(new_x[int(oldest_data_point)] >= 0)

    # Add the new sample
    sliding_window.add_element(data_point)
    new_x[int(data_point)] += 1

    new_x = new_x / sliding_window.get_num_element()
    return new_x


def update_local_vector_first_and_second_momentum(data_point, sliding_window, x):
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


def update_local_vector_concatenated_frequency_vectors(data_point, sliding_window, x):
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
