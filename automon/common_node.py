import enum
import threading
import numpy as np
import logging
from timeit import default_timer as timer
from automon.common_messages import parse_message_data_update, prepare_message_violation, MessageType, \
    parse_message_lazy_sync, prepare_message_local_vector_info, parse_message_header, messages_header_format, \
    parse_message_get_local_vector

logging = logging.getLogger(__name__)


class State(enum.Enum):
    Monitoring = 0
    WaitForSync = 1
    SuspectWaitForSync = 2


class CommonNode:

    def __init__(self, idx=0, func_to_monitor=None, x0_len=1, domain=None, max_f_val=np.inf, min_f_val=-np.inf):
        logging.info("Node " + str(idx) + " initialization: x0_len " + str(x0_len) + ", domain " + str(domain) + ", max_f_val " + str(max_f_val) + ", min_f_val " + str(min_f_val))
        self.idx = idx
        self.func_to_monitor = func_to_monitor
        self.x0_len = x0_len
        self.domain = [domain] * x0_len if domain is not None else None
        self.max_f_val = max_f_val  # The maximum value of the monitored function (if known, otherwise inf)
        self.min_f_val = min_f_val  # The minimum value of the monitored function (if known, otherwise -inf)
        self.lock = threading.Semaphore()
        CommonNode._init(self)

    def _init(self):
        self.l_thresh = 0
        self.u_thresh = 0
        self.slack = 0
        self.b_before_first_sync = True
        self.x0 = np.zeros(self.x0_len, dtype=np.float32)  # Global vector
        self.x = self.x0.copy()  # Current local vector
        self.x0_local = self.x0.copy()  # Local vector at the time of the last sync
        self.violation_origin = None
        self.state = State.Monitoring
        self.constraint_version = 0

        self.data_update_accumulated_time = 0
        self.data_update_accumulated_time_square = 0
        self.data_update_counter = 0
        self.full_sync_history = []
        self.bytes_sent = 0
        self.bytes_received = 0
        self.num_messages_sent = 0
        self.num_messages_received = 0

    def _get_point_to_check(self):
        point_to_check = self.x - self.slack
        return point_to_check

    def _inside_domain(self, x):
        # Check if the point is inside the domain.
        # If the domain is None it contains the entire sub-space and therefore, the point is always inside the domain.
        # Otherwise, the domain is a list of tuples [(min_domain_x_0,max_domain_x_0),(min_domain_x_1,max_domain_x_1),...].
        if self.domain is None:
            return True

        if not np.all(x >= np.array([min_domain for (min_domain, max_domain) in self.domain])):
            return False
        if not np.all(x <= np.array([max_domain for (min_domain, max_domain) in self.domain])):
            return False

        return True

    def _report_violation(self, violation_origin):
        self.violation_origin = violation_origin

    def _sync_common(self, x0, slack, l_thresh, u_thresh):
        self.b_before_first_sync = False
        self.x0 = x0.copy()
        self.slack = slack
        self.x0_local = self.x.copy()  # Fix local vector at sync

        self.l_thresh = l_thresh
        self.u_thresh = u_thresh

    def _lazy_sync(self, slack):
        self.slack = slack

    # Override by inherent class
    def _handle_sync_message(self, payload):
        raise NotImplementedError("To be implemented by inherent class")

    def _log_time_mean_and_std(self, counter, accumulated_time, accumulated_time_square, timer_title):
        logging.info("Node " + str(self.idx) + " " + timer_title + " counter " + str(counter))
        if counter > 0:
            mean = accumulated_time / counter
            var = (accumulated_time_square / counter) - mean ** 2
            std = np.sqrt(var)
            logging.info("Node " + str(self.idx) + " Avg " + timer_title + " time " + str(mean))
            logging.info("Node " + str(self.idx) + " Std " + timer_title + " time " + str(std))

    def get_local_vector(self):
        return self.x

    # Override by inherent class.
    # This function is called internally from set_new_data_point() to verify the new local vector.
    # It is also called by the coordinator that uses dummy node as part of the lazy sync procedure, to verify if it was
    # able to resolve violations with subset of the nodes.
    def inside_effective_safe_zone(self, x):
        raise NotImplementedError("To be implemented by inherent class")

    def set_new_data_point(self, data_point):
        start = timer()

        # Use the data point to update the local vector.
        # If needed, report violation to coordinator.
        self.x = data_point
        x = self._get_point_to_check()
        res = self.inside_effective_safe_zone(x)

        end = timer()
        self.data_update_accumulated_time += end - start
        self.data_update_accumulated_time_square += (end - start)**2
        self.data_update_counter += 1

        return res

    def dump_stats(self, test_folder):
        self._log_time_mean_and_std(self.data_update_counter, self.data_update_accumulated_time, self.data_update_accumulated_time_square, "data update")
        logging.info("Node " + str(self.idx) + " Bytes sent " + str(self.bytes_sent))
        logging.info("Node " + str(self.idx) + " Bytes received " + str(self.bytes_received))
        logging.info("Node " + str(self.idx) + " Num messages sent " + str(self.num_messages_sent))
        logging.info("Node " + str(self.idx) + " Num messages received " + str(self.num_messages_received))
        if test_folder is not None:
            with open(test_folder + "/" + self.node_name + "_node_" + str(self.idx) + "_full_sync_history.csv", 'wb') as f:
                np.savetxt(f, self.full_sync_history)

    def parse_message(self, message: bytes):
        with self.lock:
            message_out = None
            message_type, node_idx, payload_len = parse_message_header(message)
            assert (node_idx == self.idx)
            payload = message[messages_header_format.size:]

            if message_type != MessageType.DataUpdate:
                # Count only data sent by the coordinator to the node
                self.bytes_received += len(message)
                self.num_messages_received += 1

            if message_type == MessageType.GetLocalVector:
                constraint_version = parse_message_get_local_vector(payload)
                if constraint_version != self.constraint_version:
                    # Something is wrong with this request. Ignore it.
                    logging.warning("Node " + str(self.idx) + ": Got GetLocalVector message with constraint version " + str(constraint_version) + " that is different from current (" + str(self.constraint_version) + ")")
                    return None

                if self.state == State.WaitForSync:
                    # The node had already reported violation (with its local vector). Can ignore this message.
                    logging.info("Node " + str(self.idx) + ": Got GetLocalVector right after reporting violation, after " + str(self.data_update_counter) + " DataUpdate message. Ignoring.")
                else:
                    message_out = prepare_message_local_vector_info(self.idx, self.constraint_version, self.x)
                    # Should not change state here to WaitForSync. In a real distributed experiment the coordinator may
                    # ask for local vector but can decide to not use this node in lazy sync (and use only nodes with violations).
                    self.state = State.SuspectWaitForSync
            elif message_type == MessageType.LazySync:
                constraint_version, slack = parse_message_lazy_sync(payload, self.x0.shape[0])
                self.constraint_version = constraint_version
                self._lazy_sync(slack)
                self.state = State.Monitoring
            elif message_type == MessageType.Sync:
                self._handle_sync_message(payload)
                self.state = State.Monitoring
                self.full_sync_history.append(self.data_update_counter)
            elif message_type == MessageType.DataUpdate:
                if self.idx != -1 and self.state == State.WaitForSync:
                    logging.warning("Node " + str(self.idx) + ": Got DataUpdate message while state is WaitForSync after " + str(self.data_update_counter) + " data updates")
                if self.data_update_counter % 500 == 0:
                    logging.info("Node " + str(self.idx) + ": Got its " + str(self.data_update_counter) + " DataUpdate message")
                data_point = parse_message_data_update(payload, self.x0.shape[0])
                b_inside_safe_zone = self.set_new_data_point(data_point)
                if not b_inside_safe_zone:
                    self.state = State.WaitForSync
                    message_out = prepare_message_violation(self.idx, self.constraint_version, self.violation_origin, self.x)
            else:
                logging.error("Node " + str(self.idx) + ": Unexpected message type " + str(message_type))
                raise Exception

            if message_out is not None:
                self.bytes_sent += len(message_out)
                self.num_messages_sent += 1
                return message_out
