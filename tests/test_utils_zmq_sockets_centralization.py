import struct
import traceback
import zmq
import time
import threading
import logging
from timeit import default_timer as timer
import numpy as np
from automon.messages_common import MessageType, prepare_message_header
from utils.test_utils_zmq_sockets import event_monitor_client, event_monitor_server

# Use ZMQ Client-Server pattern  (https://zguide.zeromq.org/docs/chapter3/#The-Asynchronous-Client-Server-Pattern)
# Between the coordinator and the nodes. Coordinator uses ROUTER socket and the nodes use DEALER socket.


def prepare_message_local_vector_info_centralization(node_idx: int, local_vector: np.ndarray) -> bytes:
    payload = (*local_vector,)
    messages_payload_format = struct.Struct('! %dd' % local_vector.shape[0])
    message = prepare_message_header(MessageType.LocalVectorInfo, node_idx, messages_payload_format.size) + messages_payload_format.pack(*payload)
    return message


def centralization_node_data_loop(data_client, data_generator, node_idx):
    monitor = data_client.get_monitor_socket()
    t = threading.Thread(target=event_monitor_client, args=(monitor,))
    t.start()

    try:
        start_time = timer()
        logging.info('data-loop started')

        # First data update after the sliding window of the node is full
        local_vector = data_generator.get_local_vector(node_idx)
        message = prepare_message_local_vector_info_centralization(node_idx, local_vector)
        data_client.send(message)

        # For the rest of the data rounds: read data from stream and update node local vector.
        # In case of violation is will trigger sync process with the coordinator.
        while data_generator.has_next():
            # Check if the monitor thread finished
            if not t.is_alive():
                break
            local_vector, idx = data_generator.get_next_data_point()
            if idx == node_idx:
                message = prepare_message_local_vector_info_centralization(node_idx, local_vector)
                data_client.send(message)
                time.sleep(0.01)

        end = timer()
        if data_generator.has_next():
            logging.info("Node " + str(node_idx) + ": terminated by event monitor which detected the coordinator disconnected")
        logging.info("Node " + str(node_idx) + ": the test took: " + str(end - start_time) + " seconds")
    finally:
        logging.info("Node " + str(node_idx) + ": data loop ended")
        data_client.disable_monitor()
        logging.info("Node " + str(node_idx) + ": disabled event monitor")


def run_centralization_node(host, port, node_idx, data_generator):
    logging.info("Node " + str(node_idx) + ": num_iterations " + str(data_generator.get_num_iterations()) + ", data_generator state " + str(data_generator.state))

    context = zmq.Context()
    client = context.socket(zmq.DEALER)
    client.setsockopt(zmq.LINGER, 0)
    identity = '%d' % node_idx
    client.identity = identity.encode('ascii')
    client.connect('tcp://' + host + ':' + str(port))
    logging.info('Node %s started' % identity)

    try:
        # Send ready message to server socket
        client.send("ready".encode())

        # Wait for start message from the server socket
        message = client.recv()
        while message != b'start':
            message = client.recv()

        logging.info("Node " + str(node_idx) + " got start message from the coordinator")

        # Start the data loop
        centralization_node_data_loop(client, data_generator, node_idx)
        logging.info("Node " + str(node_idx) + ": main loop ended after data loop ended")
    finally:
        logging.info("Node " + str(node_idx) + ": main loop ended")
        client.close()
        logging.info("Node " + str(node_idx) + ": closed client socket")
        context.destroy()


def run_dummy_coordinator(host, port, num_nodes):
    start_time = timer()
    context = zmq.Context()
    server = context.socket(zmq.ROUTER)
    server.setsockopt(zmq.LINGER, 0)
    server.bind('tcp://' + host + ':' + str(port))

    logging.info("Coordinator server socket started")

    try:
        monitor = server.get_monitor_socket()
        t = threading.Thread(target=event_monitor_server, args=(monitor,))
        t.start()

        # Wait for ready message from all the node sockets
        b_ready_nodes = np.zeros(num_nodes, dtype=bool)
        while not np.all(b_ready_nodes):
            ident, message = server.recv_multipart()
            logging.info("Got message: " + message.decode() + " from node " + ident.decode())
            if message == b'ready':
                b_ready_nodes[int(ident)] = True

        start_time = timer()

        # After all node sockets are ready, send start signal to all the node to start the data loop
        for node_idx in range(num_nodes):
            server.send_multipart([str(node_idx).encode('ascii'), "start".encode()])

        while True:
            # Check if the monitor thread finished
            if not t.is_alive():
                break
            event = server.poll(timeout=3000)  # wait 3 seconds
            if event == 0:
                # Timeout reached before any events were queued
                pass
            else:
                # Events queued within the time limit
                ident, message = server.recv_multipart()
                if len(message) == 0:
                    logging.info("Node " + ident.decode() + " socket closed")

        # The monitor thread knows when all nodes disconnected and exits. This causes the main thread break from the loop and get here.
        server.close()
        logging.info("Coordinator stopped by the monitor thread")
    except Exception as e:
        # Exception was thrown by the coordinator
        logging.info("Coordinator stopped with an error")
        logging.info(traceback.print_exc())
        server.disable_monitor()
        t.join()
        logging.info("Coordinator : disabled event monitor")
    finally:
        server.close()
        context.destroy()
        end = timer()
        logging.info("The test took: " + str(end - start_time) + " seconds")
