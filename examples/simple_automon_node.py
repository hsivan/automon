import sys
sys.path.append('C:/Personal/AutoMonBitbucket')
import logging
import threading
import time
from timeit import default_timer as timer
import numpy as np
from automon.automon.node_common_automon import NodeCommonAutoMon
from automon.messages_common import prepare_message_data_update
from utils.utils_zmq_sockets import init_client_socket, init_client_data_socket, get_next_coordinator_message, send_message_to_coordinator


def func_inner_product(x):
    return x[:x.shape[0] // 2] @ x[x.shape[0] // 2:]


def data_loop(node_idx, host, port):
    # Open a client socket and connect to the server socket. This socket is used for reporting violations to the coordinator.
    client_data_socket = init_client_data_socket(node_idx, host, port=port)

    # Read data sample every 1 second and update the node local vector. Report violations to the coordinator.
    while True:
        start = timer()
        data = np.random.normal(loc=1, scale=0.1, size=(40,))
        message_data_update = prepare_message_data_update(node_idx, data)
        message_violation = node.parse_message(message_data_update)
        if message_violation:
            send_message_to_coordinator(client_data_socket, message_violation)
        time.sleep(1 - (timer() - start))


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = {'node_idx': 0, 'host': '127.0.0.1', 'port': 6400}  # Change node index for different nodes

    node = NodeCommonAutoMon(idx=args['node_idx'], x0_len=40, func_to_monitor=func_inner_product)
    # Open a client socket and connect to the server socket. Wait for 'start' message from the server.
    client_socket = init_client_socket(args['node_idx'], host=args['host'], port=args['port'])

    # Run the data loop in a different thread.
    threading.Thread(target=data_loop, kwargs=args).start()

    # Wait for message from the coordinator (local data requests or local constraint updates) and send the reply to the coordinator.
    while True:
        message = get_next_coordinator_message(client_socket)
        reply = node.parse_message(message)
        if reply:
            send_message_to_coordinator(client_socket, reply)
