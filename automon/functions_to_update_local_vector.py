import enum
import numpy as np


class LocalVectorUpdateType(enum.Enum):
    Average = 0
    Frequency = 1
    FirstAndSecondMomentum = 2
    ConcatenatedFrequencyVectors = 3


def get_initial_x0(local_vector_update_type: LocalVectorUpdateType, x0_len: int):
    if local_vector_update_type == LocalVectorUpdateType.Average:
        return get_initial_x0_for_average(x0_len)
    elif local_vector_update_type == LocalVectorUpdateType.Frequency:
        return get_initial_x0_for_frequency(x0_len)
    elif local_vector_update_type == LocalVectorUpdateType.FirstAndSecondMomentum:
        assert x0_len == 2
        return get_initial_x0_for_first_and_second_momentum()
    elif local_vector_update_type == LocalVectorUpdateType.ConcatenatedFrequencyVectors:
        return get_initial_x0_for_concatenated_frequency_vectors(x0_len)
    else:
        raise Exception


def get_local_vec_update_func(local_vector_update_type: LocalVectorUpdateType):
    if local_vector_update_type == LocalVectorUpdateType.Average:
        return update_local_vector_average
    elif local_vector_update_type == LocalVectorUpdateType.Frequency:
        return update_local_vector_frequency
    elif local_vector_update_type == LocalVectorUpdateType.FirstAndSecondMomentum:
        return update_local_vector_first_and_second_momentum
    elif local_vector_update_type == LocalVectorUpdateType.ConcatenatedFrequencyVectors:
        return update_local_vector_concatenated_frequency_vectors
    else:
        raise Exception


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


def get_initial_x0_for_average(x0_len):
    return np.zeros(x0_len)


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


def get_initial_x0_for_frequency(x0_len):
    return np.ones(x0_len, dtype=np.float) / x0_len


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


def get_initial_x0_for_first_and_second_momentum():
    return np.zeros(2)


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


def get_initial_x0_for_concatenated_frequency_vectors(x0_len):
    k = x0_len // 2  # Two concatenated frequency vectors
    return np.ones(x0_len, dtype=np.float) / k
