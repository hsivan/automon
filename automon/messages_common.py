import enum
import struct
import numpy as np
import logging


class ViolationOrigin(enum.Enum):
    # If the violation is caused by local vector outside the safe zone.
    SafeZone = 0
    # If the violation is caused by local vector outside the domain.
    Domain = 1
    # Faulty safe zone violations indicates the node detected violation that indicates a problem with the local constraints.
    # In that case, the coordinator should perform full sync to update the reference point, thresholds, and local constraints.
    FaultySafeZone = 2


class MessageType(enum.Enum):
    Violation = 0  # violation_origin (FaultySafe, Domain, or FaultySafeZone), local_vector
    GetLocalVector = 1  # the last constraint version that the coordinator sent to this node
    LocalVectorInfo = 2  # constraint version of the node, local_vector
    Sync = 3  # Common: global_vector, node_slack, l_thresh, u_thresh.  AutoMon: global_vector, node_slack, l_thresh, u_thresh, neighborhood_size, dc_type, dc_argument, g_func_grad_at_x0, h_func_grad_at_x0
    LazySync = 4  # node_slack
    DataUpdate = 5  # Single data point


# Message header format: (message_type:unsigned long, node_idx:long, payload_len:unsigned long)
messages_header_format = struct.Struct('! L l L')


# This function is called only by the coordinator which can get multiple messages at once (violations and local vectors) during simulation.
def message_to_message_list(messages: bytes):
    message_list = []
    unique_message_type = None
    message = messages

    while len(message) > 0:
        message_type, node_idx, payload_len = parse_message_header(message)
        payload = message[messages_header_format.size: messages_header_format.size + payload_len]

        # All messages must be of the same type
        if unique_message_type is None:
            unique_message_type = message_type  # Initialize unique_message_type with the first message in the list
        else:
            assert message_type == unique_message_type

        message_list.append((node_idx, payload))
        message = message[messages_header_format.size + payload_len:]

    return unique_message_type, message_list


def prepare_message_header(message_type: MessageType, node_idx: int, payload_len:int) -> bytes:
    # Do not print log message for DataUpdate messages as it floods the log
    if message_type != MessageType.DataUpdate:
        logging.debug("Sending message type: " + str(message_type) + " node index: " + str(node_idx) + " payload_len: " + str(payload_len))
    header = (message_type.value, node_idx, payload_len)
    return messages_header_format.pack(*header)


def parse_message_header(message: bytes):
    message_type, node_idx, payload_len = messages_header_format.unpack(message[:messages_header_format.size])
    message_type = MessageType(message_type)
    # Do not print log message for DataUpdate messages as it floods the log
    if message_type != MessageType.DataUpdate:
        logging.debug("Received message type: " + str(message_type) + " node index: " + str(node_idx) + " payload_len: " + str(payload_len))
    return message_type, node_idx, payload_len


def prepare_message_violation(node_idx: int, constraint_version: int, violation_origin: ViolationOrigin, local_vector: np.ndarray) -> bytes:
    payload = (constraint_version, violation_origin.value, *local_vector)
    messages_payload_format = struct.Struct('! L L %dd' % local_vector.shape[0])
    message = prepare_message_header(MessageType.Violation, node_idx, messages_payload_format.size) + messages_payload_format.pack(*payload)
    return message


def prepare_message_get_local_vector(node_idx: int, constraint_version: int) -> bytes:
    payload = (constraint_version,)
    messages_payload_format = struct.Struct('! L')
    message = prepare_message_header(MessageType.GetLocalVector, node_idx, messages_payload_format.size) + messages_payload_format.pack(*payload)
    return message


def prepare_message_local_vector_info(node_idx: int, constraint_version: int, local_vector: np.ndarray) -> bytes:
    payload = (constraint_version, *local_vector)
    messages_payload_format = struct.Struct('! L %dd' % local_vector.shape[0])
    message = prepare_message_header(MessageType.LocalVectorInfo, node_idx, messages_payload_format.size) + messages_payload_format.pack(*payload)
    return message


def prepare_message_sync(node_idx: int, constraint_version: int, global_vector: np.ndarray, node_slack: np.ndarray, l_thresh: float, u_thresh: float) -> bytes:
    x_len = global_vector.shape[0]
    payload = (constraint_version, *global_vector, *node_slack, l_thresh, u_thresh)
    messages_payload_format = struct.Struct('! L %dd %dd d d' % (x_len, x_len))
    message = prepare_message_header(MessageType.Sync, node_idx, messages_payload_format.size) + messages_payload_format.pack(*payload)
    return message


def prepare_message_lazy_sync(node_idx: int, constraint_version: int, node_slack: np.ndarray) -> bytes:
    payload = (constraint_version, *node_slack)
    messages_payload_format = struct.Struct('! L %dd' % node_slack.shape[0])
    message = prepare_message_header(MessageType.LazySync, node_idx, messages_payload_format.size) + messages_payload_format.pack(*payload)
    return message


def prepare_message_data_update(node_idx: int, data_point: np.ndarray) -> bytes:
    payload = (*data_point,)
    messages_payload_format = struct.Struct('! %dd' % data_point.shape[0])
    message = prepare_message_header(MessageType.DataUpdate, node_idx, messages_payload_format.size) + messages_payload_format.pack(*payload)
    return message


def parse_message_violation(payload: bytes, x0_len: int):
    messages_payload_format = struct.Struct('! L L %dd' % x0_len)
    unpacked_payload = messages_payload_format.unpack(payload)
    constraint_version = unpacked_payload[0]
    violation_origin = ViolationOrigin(unpacked_payload[1])
    local_vector = np.array(unpacked_payload[2:])
    return constraint_version, violation_origin, local_vector


def parse_message_get_local_vector(payload: bytes):
    messages_payload_format = struct.Struct('! L')
    unpacked_payload = messages_payload_format.unpack(payload)
    constraint_version = unpacked_payload[0]
    return constraint_version


def parse_message_local_vector_info(payload: bytes, x0_len: int):
    messages_payload_format = struct.Struct('! L %dd' % x0_len)
    unpacked_payload = messages_payload_format.unpack(payload)
    constraint_version = unpacked_payload[0]
    local_vector = np.array(unpacked_payload[1:])
    return constraint_version, local_vector


def parse_message_sync(payload: bytes, x0_len: int):
    messages_payload_format = struct.Struct('! L %dd %dd d d' % (x0_len, x0_len))
    unpacked_payload = messages_payload_format.unpack(payload)
    constraint_version = unpacked_payload[0]
    global_vector = np.array(unpacked_payload[1:x0_len + 1])
    node_slack = np.array(unpacked_payload[x0_len + 1:2 * x0_len + 1])
    l_thresh, u_thresh = unpacked_payload[-2], unpacked_payload[-1]
    return constraint_version, global_vector, node_slack, l_thresh, u_thresh


def parse_message_lazy_sync(payload: bytes, x0_len: int):
    messages_payload_format = struct.Struct('! L %dd' % x0_len)
    unpacked_payload = messages_payload_format.unpack(payload)
    constraint_version = unpacked_payload[0]
    node_slack = np.array(unpacked_payload[1:])
    return constraint_version, node_slack


def parse_message_data_update(payload: bytes, data_point_len: int):
    messages_payload_format = struct.Struct('! %dd' % data_point_len)
    unpacked_payload = messages_payload_format.unpack(payload)
    data_point = np.array(unpacked_payload)
    return data_point


if __name__ == "__main__":
    input_vector = np.random.randn(4)
    node_slack_org = np.random.randn(4)

    # MessageType.Violation
    message = prepare_message_violation(7, 35, ViolationOrigin.Domain, input_vector)
    message_type, node_idx, payload_len = parse_message_header(message)
    payload = message[messages_header_format.size:]
    assert node_idx == 7 and message_type == MessageType.Violation
    constraint_version, violation_origin, local_vector = parse_message_violation(payload, input_vector.shape[0])
    assert constraint_version == 35
    assert np.allclose(input_vector, local_vector)
    assert violation_origin == ViolationOrigin.Domain

    # MessageType.GetLocalVector
    constraint_version_org = 10
    message = prepare_message_get_local_vector(5, constraint_version_org)
    message_type, node_idx, payload_len = parse_message_header(message)
    payload = message[messages_header_format.size:]
    assert node_idx == 5 and message_type == MessageType.GetLocalVector
    constraint_version = parse_message_get_local_vector(payload)
    assert constraint_version == constraint_version_org

    # MessageType.LocalVectorInfo
    constraint_version_org = 20
    message = prepare_message_local_vector_info(1, constraint_version_org, input_vector)
    message_type, node_idx, payload_len = parse_message_header(message)
    payload = message[messages_header_format.size:]
    assert node_idx == 1 and message_type == MessageType.LocalVectorInfo
    constraint_version, local_vector = parse_message_local_vector_info(payload, input_vector.shape[0])
    assert np.allclose(input_vector, local_vector)
    assert constraint_version == constraint_version_org

    # MessageType.Sync
    l_thresh_org, u_thresh_org = 6.47, 18.46782
    message = prepare_message_sync(9, 106, input_vector, node_slack_org, l_thresh_org, u_thresh_org)
    message_type, node_idx, payload_len = parse_message_header(message)
    payload = message[messages_header_format.size:]
    assert node_idx == 9 and message_type == MessageType.Sync
    constraint_version, global_vector, node_slack, l_thresh, u_thresh = parse_message_sync(payload, input_vector.shape[0])
    assert constraint_version == 106
    assert np.allclose(input_vector, global_vector)
    assert np.allclose(node_slack_org, node_slack)
    assert np.allclose(l_thresh_org, l_thresh)
    assert np.allclose(u_thresh_org, u_thresh)

    # MessageType.LazySync
    message = prepare_message_lazy_sync(3, 208, node_slack_org)
    message_type, node_idx, payload_len = parse_message_header(message)
    payload = message[messages_header_format.size:]
    assert node_idx == 3 and message_type == MessageType.LazySync
    constraint_version, node_slack = parse_message_lazy_sync(payload, input_vector.shape[0])
    assert constraint_version == 208
    assert np.allclose(node_slack_org, node_slack)

    # MessageType.DataUpdate
    data_point_org = np.random.randn(5)
    message = prepare_message_data_update(2, data_point_org)
    message_type, node_idx, payload_len = parse_message_header(message)
    payload = message[messages_header_format.size:]
    assert node_idx == 2 and message_type == MessageType.DataUpdate
    data_point = parse_message_data_update(payload, data_point_org.shape[0])
    assert np.allclose(data_point_org, data_point)

    print("Done.")
