import traceback
import zmq
import time
import threading
import logging
from timeit import default_timer as timer
import numpy as np
from zmq import ContextTerminated, ZMQError
from zmq.utils.monitor import recv_monitor_message
from automon.node_common import State
from automon.messages_common import prepare_message_data_update
from automon.node_stream import NodeStream

# Use ZMQ Client-Server pattern  (https://zguide.zeromq.org/docs/chapter3/#The-Asynchronous-Client-Server-Pattern)
# Between the coordinator and the nodes. Coordinator uses ROUTER socket and the nodes use DEALER socket.


def event_monitor_client(monitor):
    try:
        while monitor.poll():
            evt = recv_monitor_message(monitor)
            if evt['event'] == zmq.EVENT_ACCEPTED:
                logging.info("Event EVENT_ACCEPTED: {}".format(evt))
            if evt['event'] == zmq.EVENT_DISCONNECTED:
                logging.info("Event EVENT_DISCONNECTED: {}".format(evt))
                break
            if evt['event'] == zmq.EVENT_MONITOR_STOPPED:
                logging.info("Event EVENT_MONITOR_STOPPED: {}".format(evt))
                break
        monitor.close()
        logging.info("Event monitor thread done")
    except (ContextTerminated, ZMQError):
        # Something went wrong
        monitor.close()
        logging.info("Event monitor thread done due to context termination")


class NodeDataLoop(threading.Thread):
    def __init__(self, context, condition, data_generator, host, port, node, node_idx, node_stream, b_single_sample_per_round):
        self.context = context
        self.condition = condition
        self.node_idx = node_idx
        self.data_generator = data_generator
        self.host = host
        self.port = port
        self.node = node
        self.node_stream = node_stream
        self.num_data_updates = 0
        self.num_detected_full_sync = 0
        self.b_single_sample_per_round = b_single_sample_per_round
        threading.Thread.__init__(self)

    def data_update(self, data_client, idx):
        # TODO: should change these values according to the network latency and coordinator full sync time
        lazy_sync_latency_seconds_first_time = 10
        lazy_sync_latency_seconds = 1.0
        full_sync_latency_seconds = 4.0
        latency_diff_between_full_and_lazy_sync = full_sync_latency_seconds - lazy_sync_latency_seconds

        if idx == self.node_idx:
            local_vector = self.node_stream.get_local_vector(self.node_idx)
            message_data_update = prepare_message_data_update(self.node_idx, local_vector)
            message_violation = self.node.parse_message(message_data_update)
            if message_violation is not None and len(message_violation) > 0:
                data_client.send(message_violation)

        self.num_data_updates += 1
        time.sleep(lazy_sync_latency_seconds_first_time + (self.num_data_updates - 1) * lazy_sync_latency_seconds + self.num_detected_full_sync * latency_diff_between_full_and_lazy_sync - (timer() - self.start_time))
        if self.node.state == State.WaitForSync or self.node.state == State.SuspectWaitForSync:
            if not self.b_single_sample_per_round and self.node.state == State.WaitForSync:
                logging.info("Node " + str(self.node_idx) + ": detected " + str(self.node.state) + " after " + str(self.num_data_updates) + " data updates")
            if self.b_single_sample_per_round:
                logging.info("Node " + str(self.node_idx) + ": detected " + str(self.node.state) + " after " + str(self.num_data_updates) + " data updates")
                self.num_detected_full_sync += 1
                time.sleep(lazy_sync_latency_seconds_first_time + (self.num_data_updates - 1) * lazy_sync_latency_seconds + self.num_detected_full_sync * latency_diff_between_full_and_lazy_sync - (timer() - self.start_time))

    def run(self):
        data_client = self.context.socket(zmq.DEALER)
        data_client.setsockopt(zmq.LINGER, 0)

        monitor = data_client.get_monitor_socket()
        t = threading.Thread(target=event_monitor_client, args=(monitor,))
        t.start()

        try:
            identity = 'data_loop-%d' % self.node_idx
            data_client.identity = identity.encode('ascii')
            data_client.connect('tcp://' + self.host + ':' + str(self.port))

            with self.condition:
                self.condition.wait()

            self.start_time = timer()
            logging.info('Node data-loop client socket %s started' % identity)

            # First data update after the sliding window of the node is full
            self.data_update(data_client, self.node_idx)

            # For the rest of the data rounds: read data from stream and update node local vector.
            # In case of violation is will trigger sync process with the coordinator.
            while self.data_generator.has_next():
                # Check if the monitor thread finished
                if not t.is_alive():
                    break
                data_point, idx = self.data_generator.get_next_data_point()
                self.node_stream.set_new_data_point(data_point, int(idx))
                if self.b_single_sample_per_round:
                    self.data_update(data_client, idx)
                else:
                    if idx == self.node_idx:
                        self.data_update(data_client, idx)

            end = timer()
            if self.data_generator.has_next():
                logging.info("Node " + str(self.node_idx) + ": terminated by event monitor which detected the coordinator disconnected")
            logging.info("Node " + str(self.node_idx) + ": the test took: " + str(end - self.start_time) + " seconds")
        finally:
            logging.info("Node " + str(self.node_idx) + ": data loop ended")
            if t.is_alive():
                data_client.disable_monitor()
            t.join()
            logging.info("Node " + str(self.node_idx) + ": disabled event monitor")
            data_client.close()
            logging.info("Node " + str(self.node_idx) + ": closed data_client socket")


def run_node(host, port, node, node_idx, data_generator, num_nodes, sliding_window_size, test_folder, b_single_sample_per_round=False):
    logging.info("Node " + str(node_idx) + ": num_nodes " + str(num_nodes) + ", num_iterations " + str(data_generator.get_num_iterations()) + ", data_generator state " + str(data_generator.state))

    node_stream = NodeStream(num_nodes, sliding_window_size, data_generator.get_data_point_len(), data_generator.get_local_vec_update_func(), initial_x0=data_generator.get_initial_x0())
    condition = threading.Condition()

    # Fill all sliding windows
    while not node_stream.all_windows_full():
        data_point, idx = data_generator.get_next_data_point()
        node_stream.set_new_data_point(data_point, int(idx))

    context = zmq.Context()
    client = context.socket(zmq.DEALER)
    client.setsockopt(zmq.LINGER, 0)
    identity = '%d' % node_idx
    client.identity = identity.encode('ascii')
    client.connect('tcp://' + host + ':' + str(port))
    logging.info('Node %s started' % identity)

    try:
        node_data_loop = NodeDataLoop(context, condition, data_generator, host, port, node, node_idx, node_stream, b_single_sample_per_round)
        node_data_loop.start()

        # Send ready message to server socket
        client.send("ready".encode())

        # Wait for start message from the server socket
        message = client.recv()
        while message != b'start':
            message = client.recv()

        logging.info("Node " + str(node_idx) + " got start message from the coordinator")
        # Signal the data loop to start
        with condition:
            condition.notifyAll()

        while True:
            # Check if the data_loop thread finished
            if not node_data_loop.is_alive():
                break
            event = client.poll(timeout=3000)  # wait 3 seconds
            if event == 0:
                # Timeout reached before any events were queued
                pass
            else:
                # Events queued within the time limit
                message = client.recv()
                if len(message) == 0:
                    logging.info("Node " + str(node_idx) + " socket closed")
                    break
                message_out = node.parse_message(message)
                if message_out is not None:
                    client.send(message_out)

        # In a regular exit the data loop thread finishes, which causes the main thread to break the loop and get here
        logging.info("Node " + str(node_idx) + ": main loop ended after data loop ended")
    finally:
        logging.info("Node " + str(node_idx) + ": main loop ended")
        with condition:
            condition.notifyAll()
        node_data_loop.join()
        client.close()
        logging.info("Node " + str(node_idx) + ": closed client socket")
        context.destroy()
        node.dump_stats(test_folder)


def event_monitor_server(monitor):
    num_connections = 0
    num_disconnections = 0
    try:
        while monitor.poll():
            evt = recv_monitor_message(monitor)
            if evt['event'] == zmq.EVENT_ACCEPTED:
                logging.info("Event EVENT_ACCEPTED: {}".format(evt))
                num_connections += 1
            if evt['event'] == zmq.EVENT_DISCONNECTED:
                logging.info("Event EVENT_DISCONNECTED: {}".format(evt))
                num_disconnections += 1
                if num_disconnections == num_connections:
                    break
            if evt['event'] == zmq.EVENT_MONITOR_STOPPED:
                logging.info("Event EVENT_MONITOR_STOPPED: {}".format(evt))
                break
        monitor.close()
        logging.info("Event monitor thread done")
    except (ContextTerminated, ZMQError):
        # In case an error occurred in the coordinator the run_coordinator stops the experiment and terminates the zeromq context.
        # This termination causes this exception in this thread.
        monitor.close()
        logging.info("Event monitor thread done due to context termination")


def run_coordinator(coordinator, port, num_nodes, test_folder):
    coordinator.b_simulation = False
    start_time = timer()
    context = zmq.Context()
    server = context.socket(zmq.ROUTER)
    server.setsockopt(zmq.LINGER, 0)
    server.bind('tcp://0.0.0.0:' + str(port))

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

                messages_out = coordinator.parse_message(message)
                for node_idx, message in messages_out:
                    server.send_multipart([str(node_idx).encode('ascii'), message])

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
        coordinator.dump_stats(test_folder)
