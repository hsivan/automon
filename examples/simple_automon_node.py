import os
import logging
from timeit import default_timer as timer
import numpy as np
from automon import AutomonNode
from automon.zmq_socket_utils import init_client_socket
from function_def import func_inner_product
logging.getLogger('automon').setLevel(logging.INFO)


def time_to_wait_for_next_sample_milliseconds(start_time, num_received_samples):
    return (num_received_samples - (timer() - start_time)) * 1000


NODE_IDX = int(os.getenv('NODE_IDX', '0'))  # Change the node index for different nodes
node = AutomonNode(idx=NODE_IDX, x0_len=40, func_to_monitor=func_inner_product)
# Open a client socket and connect to the server socket. Wait for 'start' message from the server.
client_socket = init_client_socket(NODE_IDX, host=os.getenv('HOST', '127.0.0.1'), port=6400)

# Wait for a message from the coordinator (local data requests or local constraint updates) and send the reply to the coordinator.
# Read new data samples every 1 second and update the node local vector. Report violations to the coordinator.
start = timer()
num_data_samples = 0
while True:
    if time_to_wait_for_next_sample_milliseconds(start, num_data_samples) <= 0:
        # Time to read the next data sample
        data = np.random.normal(loc=1, scale=0.1, size=(40,))
        message_violation = node.update_data(data)
        if message_violation:
            client_socket.send(message_violation)
        num_data_samples += 1
    event = client_socket.poll(timeout=time_to_wait_for_next_sample_milliseconds(start, num_data_samples))
    if event != 0:
        # Received a message from the coordinator before the timeout has reached
        message = client_socket.recv()
        reply = node.parse_message(message)
        if reply:
            client_socket.send(reply)
