import enum
import struct
import numpy as np
from automon.common_messages import MessageType, messages_header_format, prepare_message_header, parse_message_header


class DcType(enum.Enum):
    Convex = 0
    Concave = 1


def prepare_message_sync_automon(node_idx: int, constraint_version: int, global_vector: np.ndarray, node_slack: np.ndarray, l_thresh: float, u_thresh: float,
                                 neighborhood_size: float, dc_type: DcType, dc_argument: np.ndarray) -> bytes:
    x_len = global_vector.shape[0]
    if dc_argument is None:
        payload = (constraint_version, *global_vector, *node_slack, l_thresh, u_thresh, neighborhood_size, dc_type.value)
        messages_payload_format = struct.Struct('! L %dd %dd d d d L' % (x_len, x_len))
    else:
        payload = (constraint_version, *global_vector, *node_slack, l_thresh, u_thresh, neighborhood_size, dc_type.value, *dc_argument.flatten())
        messages_payload_format = struct.Struct('! L %dd %dd d d d L %dd' % (x_len, x_len, dc_argument.size))
    message = prepare_message_header(MessageType.Sync, node_idx, messages_payload_format.size) + messages_payload_format.pack(*payload)
    return message


def parse_message_sync_automon(payload: bytes, d: int):
    # dc_argument could be of size 1 (ADCD-X) or of size d^2 (ADCD-E).
    # In case that the coordinator uses ADCD-E it should only send the dc_argument once (on the first sync).
    # Therefore, a third option is that the dc_argument is omitted from the message completely (for ADCD-E sync messages
    # after the first one).
    messages_payload_format_adcd_x = struct.Struct('! L %dd %dd d d d L d' % (d, d))
    messages_payload_format_adcd_e = struct.Struct('! L %dd %dd d d d L %dd' % (d, d, d**2))
    messages_payload_format_adcd_without_dc_arg = struct.Struct('! L %dd %dd d d d L' % (d, d))

    if len(payload) == messages_payload_format_adcd_without_dc_arg.size:
        unpacked_payload = messages_payload_format_adcd_without_dc_arg.unpack(payload)
        dc_argument = None
    elif len(payload) == messages_payload_format_adcd_x.size:
        unpacked_payload = messages_payload_format_adcd_x.unpack(payload)
        dc_argument = np.array(unpacked_payload[2 * d + 5])
    else:
        unpacked_payload = messages_payload_format_adcd_e.unpack(payload)
        dc_argument = np.array(unpacked_payload[2 * d + 5:2 * d + 5 + d**2])
        dc_argument = np.reshape(dc_argument, (d, d))

    constraint_version = unpacked_payload[0]
    global_vector = np.array(unpacked_payload[1:d + 1])
    node_slack = np.array(unpacked_payload[d + 1:2 * d + 1])
    l_thresh, u_thresh = unpacked_payload[2 * d + 1], unpacked_payload[2 * d + 2]
    neighborhood_size = unpacked_payload[2 * d + 3]
    dc_type = unpacked_payload[2 * d + 4]
    dc_type = DcType(dc_type)

    return constraint_version, global_vector, node_slack, l_thresh, u_thresh, neighborhood_size, dc_type, dc_argument


if __name__ == "__main__":
    input_vector = np.random.randn(4)
    node_slack_org = np.random.randn(4)

    # MessageType.Sync (AutoMon implementation)
    l_thresh_org, u_thresh_org = 2.6, 4.9
    neighborhood_size_org = -1
    dc_argument_org = np.random.randn(input_vector.shape[0], input_vector.shape[0])

    message = prepare_message_sync_automon(3, 845, input_vector, node_slack_org, l_thresh_org, u_thresh_org, neighborhood_size_org, DcType.Convex, dc_argument_org)
    message_type, node_idx, payload_len = parse_message_header(message)
    payload = message[messages_header_format.size:]
    assert node_idx == 3 and message_type == MessageType.Sync
    constraint_version, global_vector, node_slack, l_thresh, u_thresh, neighborhood_size, dc_type, dc_argument = parse_message_sync_automon(payload, input_vector.shape[0])
    assert constraint_version == 845
    assert np.allclose(input_vector, global_vector)
    assert np.allclose(node_slack_org, node_slack)
    assert np.allclose(l_thresh_org, l_thresh)
    assert np.allclose(u_thresh_org, u_thresh)
    assert np.allclose(dc_argument_org, dc_argument)
    assert np.allclose(neighborhood_size_org, neighborhood_size)
    assert dc_type == DcType.Convex

    neighborhood_size_org = 3.4
    dc_argument_org = np.random.randn(1)
    message = prepare_message_sync_automon(5, 524, input_vector, node_slack_org, l_thresh_org, u_thresh_org, neighborhood_size_org, DcType.Concave, dc_argument_org)
    message_type, node_idx, _ = parse_message_header(message)
    payload = message[messages_header_format.size:]
    assert node_idx == 5 and message_type == MessageType.Sync
    constraint_version, global_vector, node_slack, l_thresh, u_thresh, neighborhood_size, dc_type, dc_argument = parse_message_sync_automon(payload, input_vector.shape[0])
    assert constraint_version == 524
    assert np.allclose(input_vector, global_vector)
    assert np.allclose(node_slack_org, node_slack)
    assert np.allclose(l_thresh_org, l_thresh)
    assert np.allclose(u_thresh_org, u_thresh)
    assert np.allclose(dc_argument_org, dc_argument)
    assert np.allclose(neighborhood_size_org, neighborhood_size)
    assert dc_type == DcType.Concave

    neighborhood_size_org = 1.654
    message = prepare_message_sync_automon(1, 12, input_vector, node_slack_org, l_thresh_org, u_thresh_org, neighborhood_size_org, DcType.Concave, None)
    message_type, node_idx, _ = parse_message_header(message)
    payload = message[messages_header_format.size:]
    assert node_idx == 1 and message_type == MessageType.Sync
    constraint_version, global_vector, node_slack, l_thresh, u_thresh, neighborhood_size, dc_type, dc_argument = parse_message_sync_automon(payload, input_vector.shape[0])
    assert constraint_version == 12
    assert np.allclose(input_vector, global_vector)
    assert np.allclose(node_slack_org, node_slack)
    assert np.allclose(l_thresh_org, l_thresh)
    assert np.allclose(u_thresh_org, u_thresh)
    assert dc_argument is None
    assert np.allclose(neighborhood_size_org, neighborhood_size)
    assert dc_type == DcType.Concave

    print("Done.")
