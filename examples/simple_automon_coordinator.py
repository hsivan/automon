import logging
from automon import AutomonCoordinator
from automon.zmq_socket_utils import init_server_socket, get_next_node_message, send_message_to_node
from function_def import func_inner_product
logging.getLogger('automon').setLevel(logging.INFO)

coordinator = AutomonCoordinator(num_nodes=4, func_to_monitor=func_inner_product, error_bound=2.0, d=40)
# Open a server socket. Wait for all nodes to connect and send 'start' signal to all nodes to start their data loop.
server_socket = init_server_socket(port=6400, num_nodes=4)

while True:
    msg = get_next_node_message(server_socket)
    replies = coordinator.parse_message(msg)
    for node_idx, reply in replies:
        send_message_to_node(server_socket, node_idx, reply)
