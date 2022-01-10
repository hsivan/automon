import enum
import numpy as np
import logging
from timeit import default_timer as timer
import threading
from automon.common_messages import MessageType, ViolationOrigin, parse_message_violation, parse_message_local_vector_info, \
    prepare_message_sync, prepare_message_lazy_sync, prepare_message_get_local_vector, message_to_message_list


logging = logging.getLogger(__name__)


class SlackType(enum.Enum):
    # When no slack is used each node monitors its local value x.
    NoSlack = 0
    # If using drift slack then each local node checks that x0+drift is in
    # the safe zone.
    # Drift is: x-x0_local, and x is the local vector x.
    # The node is oblivious to the slack used and just checks that x-slack is in the safe zone.
    # Therefore, the slack given to the node is x0_local-x0.
    Drift = 1


class SyncType(enum.Enum):
    # Sync all nodes.
    Eager = 0
    # Add random nodes to the set S of synced nodes, until all nodes are in the safe zone.
    LazyRandom = 1
    # Add nodes to the set S according the LRU order.
    LazyLRU = 2
    
    def is_lazy(self):
        return self != SyncType.Eager


# This class is used to separate the statistics that are collected during experiments from the core code of the coordinator.
class Statistics:

    def __init__(self):
        # Statistics that can be collected only when b_simulation is True
        self.real_function_value = []
        self.function_approximation_error = []
        self.cumulative_msg_for_broadcast_disabled = []
        self.cumulative_msg_for_broadcast_enabled = []
        self.cumulative_fallback_to_eager_sync = [0]

        # General statistics
        self.full_sync_history_times = []
        self.collect_local_vectors_latencies = []

        # Message statistics
        self.total_violations_msg_counter = 0
        self.sync_broadcast_msg_counter = 0
        self.sync_msg_counter = 0
        self.get_node_local_vector_msg_counter = 0
        self.get_node_local_vector_broadcast_msg_counter = 0
        self.node_return_local_vector_msg_counter = 0
        self.bytes_sent = 0
        self.bytes_received = 0

        # Violation statistics that can be collected only when b_simulation is True
        self.true_violations_msg_counter = 0
        self.false_global_violation_msg_counter = 0
        self.false_local_violation_msg_counter = 0
        self.missed_violations_counter = 0
        self.rounds_with_violation_counter = 0

        # Violation statistics
        self.violation_origin_outside_safe_zone = 0
        self.violation_origin_outside_domain = 0
        self.violation_origin_faulty_safe_zone = 0

        # For regression test
        self.full_sync_history = []

    def update_sync_statistics(self, f_at_global_x, f_at_x0, b_violation, b_eager_sync):
        # Keep the real function value and the error for statistics
        self.real_function_value.append(f_at_global_x)
        # The difference between f(x0) (f at the reference point from the last sync) and the real f(global_x) at the moment
        self.function_approximation_error.append(np.abs(f_at_global_x - f_at_x0))
        self.cumulative_msg_for_broadcast_enabled.append(self._total_msgs_for_enabled_broadcast())
        self.cumulative_msg_for_broadcast_disabled.append(self._total_msgs_for_disabled_broadcast())

        self.rounds_with_violation_counter += int(b_violation)
        self.cumulative_fallback_to_eager_sync.append(self.cumulative_fallback_to_eager_sync[-1] + int(b_eager_sync))

    def update_sync_messages_statistics(self, num_synced_nodes):
        # If broadcast is supported, then count single msg for the entire node group
        self.sync_broadcast_msg_counter += 1
        # Otherwise, count single messages
        self.sync_msg_counter += num_synced_nodes

    def update_get_node_local_vector_messages_statistics(self, num_asked_nodes):
        # If broadcast is supported, then count single msg for the entire node group
        self.get_node_local_vector_broadcast_msg_counter += 1
        # Otherwise, count single messages
        self.get_node_local_vector_msg_counter += num_asked_nodes

    def update_node_local_vector_info_messages_statistics(self, num_responding_nodes):
        # Update the counter that counts the responses
        self.node_return_local_vector_msg_counter += num_responding_nodes

    def update_violation_messages_statistics(self, violation_origin):
        self.total_violations_msg_counter += 1
        self.violation_origin_outside_safe_zone += int(violation_origin == ViolationOrigin.SafeZone)
        self.violation_origin_outside_domain += int(violation_origin == ViolationOrigin.Domain)
        self.violation_origin_faulty_safe_zone += int(violation_origin == ViolationOrigin.FaultySafeZone)

    def _total_msgs_for_enabled_broadcast(self):
        total_msg = self.total_violations_msg_counter + self.sync_broadcast_msg_counter + self.get_node_local_vector_broadcast_msg_counter + self.node_return_local_vector_msg_counter
        return total_msg

    def _total_msgs_for_disabled_broadcast(self):
        total_msg = self.total_violations_msg_counter + self.sync_msg_counter + self.get_node_local_vector_msg_counter + self.node_return_local_vector_msg_counter
        return total_msg

    def dump_stats(self, test_folder, coordinator_name):
        logging.info("Coordinator " + coordinator_name + " statistics:")
        logging.info("True violations msg counter " + str(self.true_violations_msg_counter))
        logging.info("False Global violations msg counter " + str(self.false_local_violation_msg_counter))
        logging.info("False Local violations msg counter " + str(self.false_local_violation_msg_counter))
        logging.info("Sync broadcast msg counter " + str(self.sync_broadcast_msg_counter))
        logging.info("Sync msg counter " + str(self.sync_msg_counter))
        logging.info("Get node statistics broadcast msg counter " + str(self.get_node_local_vector_broadcast_msg_counter))
        logging.info("Get node statistics msg counter " + str(self.get_node_local_vector_msg_counter))
        logging.info("Missed violations counter " + str(self.missed_violations_counter))
        logging.info("Rounds with violations counter " + str(self.rounds_with_violation_counter))
        logging.info("Total violations msg counter " + str(self.total_violations_msg_counter))
        logging.info("Node return statistics msg counter " + str(self.node_return_local_vector_msg_counter))
        logging.info("Total msgs broadcast enabled " + str(self._total_msgs_for_enabled_broadcast()) + ", and disabled " + str(self._total_msgs_for_disabled_broadcast()))
        logging.info("Num violations caused by local vector outside safe zone " + str(self.violation_origin_outside_safe_zone))
        logging.info("Num violations caused by local vector outside domain " + str(self.violation_origin_outside_domain))
        logging.info("Num violations caused by faulty safe zone " + str(self.violation_origin_faulty_safe_zone))
        logging.info("Bytes sent " + str(self.bytes_sent))
        logging.info("Bytes received " + str(self.bytes_received))

        logging.info("Full sync history len " + str(len(self.full_sync_history_times)))
        if len(self.full_sync_history_times) > 1:
            logging.info("Avg full sync time (ignore first time) " + str(np.mean(self.full_sync_history_times[1:])))
            logging.info("Std full sync time (ignore first time) " + str(np.std(self.full_sync_history_times[1:])))

        logging.info("Avg collect local vectors latency " + str(np.mean(self.collect_local_vectors_latencies)))
        logging.info("Std collect local vectors latency " + str(np.std(self.collect_local_vectors_latencies)))
        logging.info("Max collect local vectors latency " + str(np.max(self.collect_local_vectors_latencies, initial=0)))

        if test_folder is not None:
            with open(test_folder + "/results.txt", "a") as f:
                f.write("\n\nCoordinator " + coordinator_name + " statistics:")
                f.write("\nTrue violations " + str(self.true_violations_msg_counter))
                f.write("\nFalse Global violations " + str(self.false_global_violation_msg_counter))
                f.write("\nFalse Local violations " + str(self.false_local_violation_msg_counter))
                f.write("\nSync broadcast msg counter " + str(self.sync_broadcast_msg_counter))
                f.write("\nSync msg counter " + str(self.sync_msg_counter))
                f.write("\nGet node statistics broadcast msg counter " + str(self.get_node_local_vector_broadcast_msg_counter))
                f.write("\nGet node statistics msg counter " + str(self.get_node_local_vector_msg_counter))
                f.write("\nMissed violations counter " + str(self.missed_violations_counter))
                f.write("\nRounds with violations counter " + str(self.rounds_with_violation_counter))
                f.write("\nTotal violations msg counter " + str(self.total_violations_msg_counter))
                f.write("\nNode return statistics msg counter " + str(self.node_return_local_vector_msg_counter))
                f.write("\nTotal msgs broadcast enabled " + str(self._total_msgs_for_enabled_broadcast()) + ", and disabled " + str(self._total_msgs_for_disabled_broadcast()))
                f.write("\nNum violations caused by local vector outside safe zone " + str(self.violation_origin_outside_safe_zone))
                f.write("\nNum violations caused by local vector outside domain " + str(self.violation_origin_outside_domain))
                f.write("\nNum violations caused by faulty safe zone " + str(self.violation_origin_faulty_safe_zone))
                f.write("\nBytes sent " + str(self.bytes_sent))
                f.write("\nBytes received " + str(self.bytes_received))
                f.write("\nFull sync history len " + str(len(self.full_sync_history_times)))
                if len(self.full_sync_history_times) > 1:
                    f.write("\nAvg full sync time (ignore first time) " + str(np.mean(self.full_sync_history_times[1:])))
                    f.write("\nStd full sync time (ignore first time) " + str(np.std(self.full_sync_history_times[1:])))
                f.write("\nAvg collect local vectors latency " + str(np.mean(self.collect_local_vectors_latencies)))
                f.write("\nStd collect local vectors latency " + str(np.std(self.collect_local_vectors_latencies)))
                f.write("\nMax collect local vectors latency " + str(np.max(self.collect_local_vectors_latencies, initial=0)))

            # Write "over time" arrays to files. Ignore the first value that is of the initial sync (and initial x0).
            file_prefix = test_folder + "/" + coordinator_name
            with open(file_prefix + "_real_function_value.csv", 'wb') as f:
                np.savetxt(f, self.real_function_value)
            with open(file_prefix + "_function_approximation_error.csv", 'wb') as f:
                np.savetxt(f, self.function_approximation_error)
            with open(file_prefix + "_cumulative_msgs_broadcast_enabled.csv", 'wb') as f:
                np.savetxt(f, self.cumulative_msg_for_broadcast_enabled)
            with open(file_prefix + "_cumulative_msgs_broadcast_disabled.csv", 'wb') as f:
                np.savetxt(f, self.cumulative_msg_for_broadcast_disabled)
            with open(file_prefix + "_cumulative_fallback_to_eager_sync.csv", 'wb') as f:
                np.savetxt(f, self.cumulative_fallback_to_eager_sync)
            with open(file_prefix + "_full_sync_times.csv", 'wb') as f:
                np.savetxt(f, self.full_sync_history_times)

    def get_msg_counters(self):
        return [self.true_violations_msg_counter,
                self.false_global_violation_msg_counter,
                self.false_local_violation_msg_counter,
                self.sync_broadcast_msg_counter,
                self.sync_msg_counter,
                self.get_node_local_vector_broadcast_msg_counter,
                self.get_node_local_vector_msg_counter,
                self.missed_violations_counter,
                self.rounds_with_violation_counter,
                self.total_violations_msg_counter,
                self.node_return_local_vector_msg_counter,
                self._total_msgs_for_enabled_broadcast(),
                self._total_msgs_for_disabled_broadcast()]


class State(enum.Enum):
    # From Idle state can move to LazySync or FullSync
    Idle = 0
    # From LazySync can move to Idle (if was able to resolve violations) or to FullSync (if failed to resolve violations).
    LazySync = 1
    # From FullSync moves to Idle after receiving LocalVectorInfo messages from all the nodes.
    FullSync = 2


class CommonCoordinator:
    
    def __init__(self, verifier, num_nodes, error_bound=2, slack_type=SlackType.Drift, sync_type=SyncType.Eager,
                 lazy_sync_max_S=0.5, b_violation_strict=True, coordinator_name="Common"):
        self.coordinator_name = coordinator_name
        # Relevant only for simulation. Indicates whether this type of coordinator tolerates false negative events (missed violations).
        self.b_violation_strict = b_violation_strict
        # Flag that indicates if the current run is simulation or not. The test manager sets to True, after initialization, if running as simulation.
        self.b_simulation = False

        self.lock = threading.Semaphore()
        
        self.verifier = verifier  # Node that is used in lazy sync (to verify constraints) and for violation statistics.
        self.func_to_monitor = verifier.func_to_monitor
        self.error_bound = error_bound
        self.slack_type = slack_type
        self.sync_type = sync_type
        assert(not (slack_type == SlackType.NoSlack and sync_type.is_lazy()))
        self.lazy_sync_max_S = lazy_sync_max_S
        self.num_nodes = num_nodes

        CommonCoordinator._init(self)
        logging.info(self.coordinator_name + " coordinator initialization: x0_len " + str(self.x0_len) + ", error_bound " + str(error_bound) + ", num_nodes " + str(num_nodes) +
                     ", slack_type " + str(slack_type) + ", sync_type " + str(sync_type) + ", lazy_sync_max_S " + str(lazy_sync_max_S))

    def _init(self):
        self.iteration = 0
        self.state = State.Idle
        self.indices_of_nodes_asked_for_local_vector = []
        self.verifier._init()
        self.x0 = self.verifier.get_local_vector()
        self.x0_len = self.x0.shape[0]
        self.u_thresh = 0
        self.l_thresh = 0
        self.b_faulty_safe_zone = False
        self.b_violation = False
        self.b_eager_sync = False

        # Nodes
        self.nodes_x0_local = np.zeros((self.num_nodes, self.x0_len))
        # Indicates if node sent its local vector in the current iteration.
        # It could be due to violation msg from this node, or during lazy sync process.
        # It tells the coordinator, during eager sync for example, that it does not need to collect the local vector from this node.
        self.b_nodes_have_updated_local_vector = np.zeros(self.num_nodes, dtype=bool)
        self.nodes_slack = np.zeros((self.num_nodes, self.x0_len))
        self.b_nodes_have_violation = np.zeros(self.num_nodes, dtype=bool)
        self.b_nodes_have_violation_prev_iteration = self.b_nodes_have_violation.copy()
        self.nodes_lazy_lru_sync_counter = np.zeros(self.num_nodes)
        # Keep for each node its constraint version. After eager sync all the nodes should hold the latest version.
        # After lazy sync only the nodes in S should hold the latest version and the rest of the nodes an older version.
        # Messages between the coordinator and the nodes include these versions.
        self.nodes_constraint_version = np.zeros(self.num_nodes, dtype=int)

        # Collect statistics during experiment
        self.statistics = Statistics()

    def _global_vector_inside_admissible_region(self):
        # Check if the global x, which is the one in the verifier (which uses no slack) is inside the admissible region.
        # This verification is used for statistics such as number of true violations, false local violations, false global violations, etc.
        global_x = self.verifier.get_local_vector()
        f_at_x = self.func_to_monitor(global_x)
        return self.l_thresh <= f_at_x <= self.u_thresh

    def _global_vector_inside_effective_safe_zone(self):
        # Check if the global x, which is the one in the verifier (which uses no slack) is inside the effective safe zone
        # (inside domain, bounds, safe zone).
        # This verification is used for statistics such as number of true violations, false local violations, false global violations, etc.
        global_x = self.verifier.get_local_vector()
        return self.verifier.inside_effective_safe_zone(global_x)
    
    def _log_violation_type(self, node_idx):
        # Find the type and origin of the violation and write it to log file and update statistics

        b_inside_admissible_region = self._global_vector_inside_admissible_region()
        b_inside_safe_zone = self._global_vector_inside_effective_safe_zone()
        # This is a "true" violation if global x is not in the admissible region
        b_true_violation = not b_inside_admissible_region
        # This is a "false global" violation if global x is not in the safe zone but inside the admissible region
        b_false_global_violation = not b_inside_safe_zone and b_inside_admissible_region
        # This is a "false local" violation if global x is inside the safe zone
        b_false_local_violation = b_inside_safe_zone
        
        if self.b_violation_strict:
            assert(b_true_violation + b_false_global_violation + b_false_local_violation == 1)
        else:
            # Do not assert, just log the error. This is needed in AutomonCoordinator and RlvCoordinator, when this error can happen.
            if b_true_violation + b_false_global_violation + b_false_local_violation != 1:
                logging.warning("Iteration " + str(self.iteration) + ": b_true_violation " + str(b_true_violation) + ", b_false_global_violation " + str(b_false_global_violation) + ", b_false_local_violation " + str(b_false_local_violation))

        self.statistics.true_violations_msg_counter += int(b_true_violation)
        self.statistics.false_global_violation_msg_counter += int(b_false_global_violation)
        self.statistics.false_local_violation_msg_counter += int(b_false_local_violation)

        violation_type_str = ""
        if b_true_violation:
            violation_type_str = "True Violation"
        if b_false_global_violation:
            violation_type_str = "False Global Violation" if violation_type_str == "" else violation_type_str + " and False Global Violation"
        if b_false_local_violation:
            violation_type_str = "False Local Violation" if violation_type_str == "" else violation_type_str + " and False Global Violation"
        logging.info("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " notify " + violation_type_str)

    def _notify_violation(self, node_idx, violation_origin):
        self.b_nodes_have_violation[node_idx] = True
        self.b_violation = True  # For statistics of iterations with violations

        if self.b_simulation:
            self._log_violation_type(node_idx)

        if violation_origin == ViolationOrigin.FaultySafeZone:
            # Should perform full sync to resolve the issue
            logging.warning("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " notify faulty safe zone violation. Trigger full sync.")
            self.b_faulty_safe_zone = True

    def _prepare_message_get_local_vector_for_node_group(self, nodes_indices):
        messages_out = []

        # Get stats from nodes with outdated statistics
        indices_of_nodes_asked_for_local_vector = [node_idx for node_idx in nodes_indices if not self.b_nodes_have_updated_local_vector[node_idx]]
        # Wait for local vectors of these outdated nodes
        self.indices_of_nodes_asked_for_local_vector = indices_of_nodes_asked_for_local_vector

        if len(indices_of_nodes_asked_for_local_vector) > 0:
            logging.info("Iteration " + str(self.iteration) + ": Coordinator about to ask " + str(len(indices_of_nodes_asked_for_local_vector)) + " nodes for statistics. Nodes " + str(indices_of_nodes_asked_for_local_vector))
            self.statistics.update_get_node_local_vector_messages_statistics(len(indices_of_nodes_asked_for_local_vector))
        
        for node_idx in indices_of_nodes_asked_for_local_vector:
            logging.info("Iteration " + str(self.iteration) + ": Coordinator asks node " + str(node_idx) + " for statistics")
            message_out = prepare_message_get_local_vector(node_idx, self.nodes_constraint_version[node_idx])
            messages_out.append((node_idx, message_out))

        return messages_out

    def _update_local_vector_info(self, node_idx, x):
        self.nodes_x0_local[node_idx] = x
        self.b_nodes_have_updated_local_vector[node_idx] = True
        if node_idx in self.indices_of_nodes_asked_for_local_vector:
            self.indices_of_nodes_asked_for_local_vector.remove(node_idx)

    def _eager_sync(self):
        # Collect all local statistic vectors from all the nodes and compute new x0 and local constrains.
        # Set all nodes with the new x0 value and constraints
        messages_out = self._prepare_message_get_local_vector_for_node_group(list(range(self.num_nodes)))
        return messages_out

    def _finish_eager_sync(self):
        start = timer()
        self.b_eager_sync = True

        # Already collect all local statistic vectors from all the nodes.
        # Compute new x0 and local constrains.
        # Set all nodes with the new x0 value and constraints.
        new_x0, _ = self._evaluate_x0_and_slack(list(range(self.num_nodes)))

        if self.b_simulation:
            # Sanity check: verify that new_x0 is the same one as the verifier x (which is the global vector)
            global_x = self.verifier.get_local_vector()
            assert (np.all(global_x - new_x0 < 1e-10))
        else:
            # This action is not required as the global vector x of the verifier is not used in a real distributed experiment.
            self.verifier.x = new_x0.copy()
        logging.info("Iteration " + str(self.iteration) + ": About to sync the value " + str(new_x0))

        self.x0 = new_x0
        # Updating the thresholds to make sure that that the new x0 is inside the safe zone.
        self._update_l_u_threshold()

        # Update the slacks to all nodes, and sync all nodes
        self._allocate_slacks(self.x0, list(range(self.num_nodes)))
        messages_out = self._sync_nodes(list(range(self.num_nodes)), sync_type="full")

        # Sync also verifier. Since verifier.x equals new_x0, no slack is ever needed.
        self._sync_verifier()
        # new_x0 must be inside the safe zone. We can make sure by checking that verifier.x
        # is inside the safe zone since verifier.x equals new_x0.
        assert (self._global_vector_inside_effective_safe_zone())

        self.b_faulty_safe_zone = False

        end = timer()

        self.statistics.full_sync_history.append((self.iteration, new_x0))  # For testing: keep the iteration and the new x0
        self.statistics.full_sync_history_times.append(end - start)
        if self.iteration == 0:
            # This is the first full sync after windows of all nodes are full. Should ignore all violations up until now.
            self.statistics.total_violations_msg_counter = 0
            self.statistics.violation_origin_outside_safe_zone = 0
            self.statistics.violation_origin_outside_domain = 0

        return messages_out
        
    def _lazy_sync(self):
        b_eager_sync_fallback = False
        S_max_size = np.round(self.lazy_sync_max_S * self.num_nodes)

        # Before asking collecting the local vectors of extra nodes, try first to resolve violations with the nodes with violations.
        # This is only relevant if a violation was reported after the previous call to _lazy_sync().
        if not np.alltrue(self.b_nodes_have_violation_prev_iteration == self.b_nodes_have_violation):
            S = np.nonzero(self.b_nodes_have_violation)[0]
            if len(S) <= S_max_size:
                S_x0, S_slack = self._evaluate_x0_and_slack(S)
                if self.verifier.inside_effective_safe_zone(S_x0 - S_slack):
                    logging.info("Iteration " + str(self.iteration) + ": Resolved violations only with violating nodes")
                    if len(S) == 1:
                        logging.error("Iteration " + str(self.iteration) + ": Used a single node in lazy sync")
                        raise Exception
                    self.b_nodes_have_updated_local_vector = self.b_nodes_have_violation.copy()
                    # The violation is resolved using the nodes in S. No need to ask for more local vectors and can move to _finish_lazy_sync step.
                    return [], b_eager_sync_fallback

        self.b_nodes_have_violation_prev_iteration = self.b_nodes_have_violation.copy()

        S = np.nonzero(self.b_nodes_have_updated_local_vector)[0]

        # Now try to resolve violations with the nodes that provide their local vectors (due to violations or as part of LOCAL_VECTOR_INFO message)
        if len(S) <= S_max_size:
            S_x0, S_slack = self._evaluate_x0_and_slack(S)
            if self.verifier.inside_effective_safe_zone(S_x0 - S_slack):
                # The violation is resolved using the nodes in S. No need to ask for more local vectors and can move to _finish_lazy_sync step.
                return [], b_eager_sync_fallback

        # Could not resolve violations with the nodes that provide their local vectors.

        if len(S) >= S_max_size:
            logging.info("Iteration " + str(self.iteration) + ": Fallback to eager sync from lazy sync !!!!!!!!!!!!!!!!!!")
            messages_out = self._eager_sync()
            b_eager_sync_fallback = True
            # Reset the LRU counters of all nodes
            self.nodes_lazy_lru_sync_counter = np.zeros(self.num_nodes)
            return messages_out, b_eager_sync_fallback

        # Add nodes to S until the convex combination of the vectors (x_i-s_i) is in the safe zone

        S_not = np.nonzero(np.logical_not(self.b_nodes_have_updated_local_vector))[0]
        if self.sync_type == SyncType.LazyRandom:
            # Arrange S_not (the nodes without violations) in random order
            np.random.shuffle(S_not)
        if self.sync_type == SyncType.LazyLRU:
            # Arrange S_not (the nodes without violations) according to LRU
            S_not_lru_counters = self.nodes_lazy_lru_sync_counter[S_not]
            S_not = S_not[S_not_lru_counters.argsort()]

        node_idx = S_not[0]
        messages_out = self._prepare_message_get_local_vector_for_node_group([node_idx])
        return messages_out, b_eager_sync_fallback

    def _finish_lazy_sync(self):
        S = np.nonzero(self.b_nodes_have_updated_local_vector)[0]
        logging.info("Iteration " + str(self.iteration) + ": Used " + str(len(S)) + " nodes in lazy sync. Nodes " + str(S))
        S_x0, S_slack = self._evaluate_x0_and_slack(S)
        # Allocate slack and sync nodes
        self._allocate_slacks(S_x0 - S_slack, S)
        messages_out = self._sync_nodes(S, sync_type="lazy")
        # Update the LRU counters of the nodes in S
        self.nodes_lazy_lru_sync_counter[S] += 1
        return messages_out

    def _check_missed_violations(self):
        # Check for missed violations (false negative). It is only possible to have missed violations in AutomonCoordinator
        # in case the coordinator didn't find the real min/max eigenvalue, and in RlvCoordinator.
        # In that case there is violation of the admissible region, but no violation from any of the nodes.
        # We check it here, since this function is called after each round of set_new_data_point() for all the nodes.
        if (not np.any(self.b_nodes_have_violation)) and (not self._global_vector_inside_admissible_region()):
            self.statistics.missed_violations_counter += 1
            if self.b_violation_strict:
                logging.error("Iteration " + str(self.iteration) + ": Found true violation without any node violation when running in strict mode.")
                raise Exception
            logging.warning("Iteration " + str(self.iteration) + ": Found true violation without any node violation.")

    # Override by inherent class. The specific coordinator specifies here its special condition for full sync.
    # By default, there is no special condition for eager sync and the coordinator uses lazy sync and falls to full sync when resolving violation fails.
    def _is_eager_sync_required(self):
        return False

    def _resolve_violation(self):
        b_eager_sync = True

        if self.b_faulty_safe_zone:
            messages_out = self._eager_sync()
        elif self._is_eager_sync_required():
            messages_out = self._eager_sync()
        elif self.sync_type == SyncType.Eager:
            messages_out = self._eager_sync()
        elif self.sync_type.is_lazy():
            messages_out, b_eager_sync = self._lazy_sync()  # Returns indication if there was a fallback to eager sync or not
        else:
            logging.error("Iteration " + str(self.iteration) + ": Unexpected sync type " + str(self.sync_type))
            raise Exception

        return messages_out, b_eager_sync

    def _evaluate_x0_and_slack(self, nodes_indices):
        x0 = np.zeros(self.x0_len)
        slack = np.zeros(self.x0_len)

        for node_idx in nodes_indices:
            x0 += self.nodes_x0_local[node_idx]
            slack += self.nodes_slack[node_idx]

        x0 = x0 / len(nodes_indices)
        slack = slack / len(nodes_indices)

        return x0, slack

    def _allocate_slacks(self, x0, nodes_indices):
        for node_idx in nodes_indices:
            slack = np.zeros_like(x0)  # self.slack_type == SlackType.NoSlack
            if self.slack_type == SlackType.Drift:
                slack = self.nodes_x0_local[node_idx] - x0
            self.nodes_slack[node_idx] = slack
        
        assert(np.isclose(np.sum(self.nodes_slack), 0))
        
    def _sync_nodes(self, nodes_indices, sync_type="full"):
        messages_out = []
        logging.info("Iteration " + str(self.iteration) + ": Coordinator about to sync " + str(len(nodes_indices)) + " nodes. Nodes " + str(nodes_indices))
        self.statistics.update_sync_messages_statistics(len(nodes_indices))
                
        for node_idx in nodes_indices:
            logging.info("Iteration " + str(self.iteration) + ": Coordinator syncs node " + str(node_idx))
            message_out = self._sync_node(node_idx, sync_type)
            messages_out.append((node_idx, message_out))
            self.b_nodes_have_violation[node_idx] = False

        # After sync there shouldn't be nodes with violations
        assert not np.any(self.b_nodes_have_violation)
        self.b_nodes_have_violation_prev_iteration = self.b_nodes_have_violation.copy()

        return messages_out

    # Override by inherent class if sync requires additional parameters
    def _sync_verifier(self):
        # Since verifier.x equals new_x0, no slack is ever needed.
        self.verifier.sync(self.x0, np.zeros_like(self.x0), self.l_thresh, self.u_thresh)

    # Override by inherent class if sync requires additional parameters
    def _sync_node(self, node_idx, sync_type="full"):
        self.nodes_constraint_version[node_idx] = self.iteration
        if sync_type == "full":
            message_out = prepare_message_sync(node_idx, self.nodes_constraint_version[node_idx], self.x0, self.nodes_slack[node_idx], self.l_thresh, self.u_thresh)
        else:
            message_out = prepare_message_lazy_sync(node_idx, self.nodes_constraint_version[node_idx], self.nodes_slack[node_idx])
        return message_out
    
    def _update_l_u_threshold(self):
        f = self.func_to_monitor(self.x0)
        self.l_thresh = f - self.error_bound
        self.u_thresh = f + self.error_bound
        logging.info("Iteration " + str(self.iteration) + ": About to sync the thresholds " + str(self.l_thresh) + "," + str(self.u_thresh))

    # This function should be called after every data round by the test util loop (in a simulation, not in a real distributed experiment). This is for statistics only.
    def update_statistics(self):
        self.iteration += 1
        self._check_missed_violations()

        self.statistics.update_sync_statistics(self.func_to_monitor(self.verifier.get_local_vector()),
                                               self.func_to_monitor(self.x0), self.b_violation, self.b_eager_sync)
        self.b_violation = False
        self.b_eager_sync = False

    def _handle_violation_message(self, message_list):
        num_updates = 0

        for node_idx, payload in message_list:
            constraint_version, violation_origin, local_vector = parse_message_violation(payload, self.x0_len)
            self.statistics.update_violation_messages_statistics(violation_origin)
            if constraint_version != self.nodes_constraint_version[node_idx]:
                logging.warning("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " reported violation " + str(violation_origin) + " with an old constraint version " + str(constraint_version) + " (current is " + str(self.nodes_constraint_version[node_idx]) + "). Ignoring.")
                continue

            if self.state == State.Idle:
                self.start_collecting_local_vectors = timer()
                if not self.b_simulation:
                    # TODO: remove. This sleep adds latency that simulates the network latency. This sleep after the first violation in a sync round
                    # enables all the nodes to receive their data and update their local vectors in this data update round, before the coordinator
                    # asks for their local vectors as part of the sync process.
                    # In a real distributed experiment the network latency should be enough (under the assumption that all the nodes receive their
                    # data at about the same time in each data round).
                    #time.sleep(0.02)  # 20 milliseconds
                    pass

            logging.info("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " notify violation " + str(violation_origin) + " with constraint version " + str(constraint_version))
            if self.b_nodes_have_violation[node_idx]:
                logging.error("Iteration " + str(self.iteration) + ": Got violation from node " + str(node_idx) + " when there is a pending violation for this node")
                raise Exception
            self._notify_violation(node_idx, violation_origin)
            self._update_local_vector_info(node_idx, local_vector)
            num_updates += 1

        return num_updates

    def _handle_local_vector_info_message(self, message_list):
        num_updates = 0

        for node_idx, payload in message_list:
            self.statistics.update_node_local_vector_info_messages_statistics(1)
            constraint_version, local_vector = parse_message_local_vector_info(payload, self.x0_len)
            # First, check if the iteration number in the message equals self.iteration. If not, the message is from
            # old iteration and should be ignored.
            if constraint_version != self.nodes_constraint_version[node_idx]:
                logging.warning("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " returns to coordinator with statistics with an old constraint version " + str(constraint_version) + " (current is " + self.nodes_constraint_version[node_idx] + "). Ignoring.")
                continue
            # Second, check if the local vector of this node was already updated. It can happen if the coordinator
            # asked for this node's local vector as part of LazySync but before it got the vector from the node,
            # the node had already reported violation (with its local vector) to the coordinator.
            # In that case, do nothing.
            if node_idx not in self.indices_of_nodes_asked_for_local_vector:
                logging.info("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " returns to coordinator with statistics, but vector was already updated")
                continue
            logging.info("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " returns to coordinator with statistics")
            self._update_local_vector_info(node_idx, local_vector)
            num_updates += 1

        return num_updates

    def _state_machine(self, message_type, message_list):
        messages_out = []

        num_updates = self._handle_violation_message(message_list) if message_type == MessageType.Violation else self._handle_local_vector_info_message(message_list)
        if num_updates == 0:
            return messages_out

        # If len(self.indices_of_nodes_asked_for_local_vector) > 0, the coordinator must wait for the rest of the nodes in indices_of_nodes_asked_for_local_vector list
        # to send their local vectors. Otherwise, it can try to move to the next state.

        if len(self.indices_of_nodes_asked_for_local_vector) == 0 and not self.state == State.FullSync:
            # All the nodes in indices_of_nodes_asked_for_local_vector list sent their local vectors back to the coordinator.
            # If state is Idle calling to self._resolve_violation() starts the sync process, lazy or eager.
            # If state is FullSync then calling to self._resolve_violation() does nothing and returns empty message (so just skip the call in this case to prevent confusing logging).
            # If state is Idle or LazySync then calling to self._resolve_violation() asks for the next nodes for their local vectors.
            messages_out, b_eager_sync = self._resolve_violation()
            if b_eager_sync:
                self.state = State.FullSync
            else:
                self.state = State.LazySync

        # Calling to self._resolve_violation() may change self.indices_of_nodes_asked_for_local_vector, therefore, must check again for its length.

        if len(self.indices_of_nodes_asked_for_local_vector) == 0:
            self.statistics.collect_local_vectors_latencies.append(timer() - self.start_collecting_local_vectors)

            if self.state == State.FullSync:
                messages_out = self._finish_eager_sync()
            elif self.state == State.LazySync:
                messages_out = self._finish_lazy_sync()
            self.state = State.Idle
            self.b_nodes_have_updated_local_vector = np.zeros(self.num_nodes, dtype=bool)
            if not self.b_simulation:
                # In a real distributed experiment the iterations are the sync rounds, and every sync round ends here, with a call to finish_sync().
                # In simulation, however, the iterations are the data update rounds, and iteration increase happens in update_statistics()
                # that is called by the test manager even if no violation occurred during this data update round.
                self.iteration += 1

        return messages_out

    def dump_stats(self, test_folder):
        self.statistics.dump_stats(test_folder, self.coordinator_name)
        return self.statistics.full_sync_history, self.statistics.get_msg_counters()

    # For compatibility with both simulation and real distributed experiment (that uses messages), this method is the
    # only entry point of the coordinator (except dump_stats function that is called directly).
    def parse_message(self, messages: bytes):
        with self.lock:
            self.statistics.bytes_received += len(messages)
            message_type, message_list = message_to_message_list(messages)

            if message_type == MessageType.Violation or message_type == MessageType.LocalVectorInfo:
                messages_out = self._state_machine(message_type, message_list)
            else:
                logging.error("Iteration " + str(self.iteration) + ": Unexpected message type " + str(message_type))
                raise Exception

            for _, message_out in messages_out:
                self.statistics.bytes_sent += len(message_out)

            return messages_out
