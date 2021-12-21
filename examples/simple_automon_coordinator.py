import sys
import logging
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.automon.node_common_automon import NodeCommonAutoMon
from automon_utils.utils_zmq_sockets import init_server_socket, get_next_node_message, send_message_to_node


def func_inner_product(x):
    return x[:x.shape[0] // 2] @ x[x.shape[0] // 2:]


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Create dummy node for the coordinator that uses it in the process of resolving violations.
    verifier = NodeCommonAutoMon(idx=-1, x0_len=40, func_to_monitor=func_inner_product)
    coordinator = CoordinatorAutoMon(verifier, num_nodes=4, error_bound=0.5)
    # Open server socket. Wait for all nodes to connect and send 'start' signal to all nodes to start their data loop.
    server_socket = init_server_socket(port=6400, num_nodes=4)

    while True:
        msg = get_next_node_message(server_socket)
        replies = coordinator.parse_message(msg)
        for node_idx, reply in replies:
            send_message_to_node(server_socket, node_idx, reply)
