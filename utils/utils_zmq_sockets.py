import traceback
import zmq
import logging

# Use ZMQ Client-Server pattern  (https://zguide.zeromq.org/docs/chapter3/#The-Asynchronous-Client-Server-Pattern)
# Between the coordinator and the nodes. Coordinator uses ROUTER socket and the nodes use DEALER socket.


def init_client_socket(node_idx, host='127.0.0.1', port=6400):
    context = zmq.Context()
    client_socket = context.socket(zmq.DEALER)
    client_socket.setsockopt(zmq.LINGER, 0)
    identity = '%d' % node_idx
    client_socket.identity = identity.encode('ascii')
    client_socket.connect('tcp://' + host + ':' + str(port))
    logging.info('Node %s started' % identity)

    try:
        # Send ready message to server socket
        client_socket.send("ready".encode())

        # Wait for start message from the server socket
        message = client_socket.recv()
        while message != b'start':
            message = client_socket.recv()

        logging.info("Node " + str(node_idx) + " got start message from the coordinator")
    except Exception as e:
        logging.info(traceback.print_exc())
    return client_socket


def init_client_data_socket(node_idx, host='127.0.0.1', port=6400):
    context = zmq.Context()
    client_socket = context.socket(zmq.DEALER)
    client_socket.setsockopt(zmq.LINGER, 0)
    identity = 'data_loop-%d' % node_idx
    client_socket.identity = identity.encode('ascii')
    client_socket.connect('tcp://' + host + ':' + str(port))
    logging.info('Node %s started' % identity)
    return client_socket


def init_server_socket(port=6400, num_nodes=10):
    # Opens server socket. Waits for all the nodes to connect and then sends 'start' signal to all the nodes to start the data loop
    context = zmq.Context()
    server_socket = context.socket(zmq.ROUTER)
    server_socket.setsockopt(zmq.LINGER, 0)
    server_socket.bind('tcp://0.0.0.0:' + str(port))

    logging.info("Coordinator server socket started")

    try:
        # Wait for ready message from all the node sockets
        b_ready_nodes = [False] * num_nodes
        while not sum(b_ready_nodes) == num_nodes:
            ident, message = server_socket.recv_multipart()
            logging.info("Got message: " + message.decode() + " from node " + ident.decode())
            if message == b'ready':
                b_ready_nodes[int(ident)] = True

        # After all node sockets are ready, send start signal to all the nodes to start the data loop
        for node_idx in range(num_nodes):
            server_socket.send_multipart([str(node_idx).encode('ascii'), "start".encode()])
    except Exception as e:
        logging.info(traceback.print_exc())
    return server_socket


def get_next_node_message(server_socket):
    ident, message = server_socket.recv_multipart()
    return message


def send_message_to_node(server_socket, node_idx, message):
    server_socket.send_multipart([str(node_idx).encode('ascii'), message])


def get_next_coordinator_message(client_socket):
    message = client_socket.recv()
    return message


def send_message_to_coordinator(client_socket, message):
    client_socket.send(message)
