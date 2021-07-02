import enum
import numpy as np
import logging
from timeit import default_timer as timer


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
    
    def is_lazy(sync_type):
        return sync_type != SyncType.Eager


class ViolationOrigin(enum.Enum):
    # If the violation is caused by local vector outside the safe zone.
    SafeZone = 0
    # If the violation is caused by local vector outside the domain.
    Domain = 1
    # Faulty safe zone violations indicates the node detected violation that indicates a problem with the local constraints.
    # In that case, the coordinator should perform full sync to update the reference point, thresholds, and local constraints.
    FaultySafeZone = 2


# This class is used to separate the statistics that are collected during experiments from the core code of the coordinator.
class Statistics:

    def __init__(self):
        self.real_function_value = []
        self.function_approximation_error = []
        self.cumulative_msg_for_broadcast_disabled = []
        self.cumulative_msg_for_broadcast_enabled = []
        self.cumulative_fallback_to_eager_sync = [0]
        self.full_sync_history_times = []

        # Message statistics
        self.true_violations_msg_counter = 0
        self.false_global_violation_msg_counter = 0
        self.false_local_violation_msg_counter = 0
        self.total_violations_msg_counter = 0
        self.sync_broadcast_msg_counter = 0
        self.sync_msg_counter = 0
        self.get_node_local_vector_msg_counter = 0
        self.get_node_local_vector_broadcast_msg_counter = 0
        self.node_return_local_vector_msg_counter = 0

        # Violation statistics
        self.miss_violations_counter = 0
        self.rounds_with_violation_counter = 0
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

    def update_sync_messages_statistics(self, iteration, num_synced_nodes):
        if iteration > 0:  # Ignore the first sync in iteration 0
            # If broadcast is supported, then count single msg for the entire node group
            self.sync_broadcast_msg_counter += 1
            # Otherwise, count single messages
            self.sync_msg_counter += num_synced_nodes

    def update_node_local_vector_messages_statistics(self, iteration, num_asked_nodes):
        if iteration > 0:  # Ignore the first sync in iteration 0
            # If broadcast is supported, then count single msg for the entire node group
            self.get_node_local_vector_broadcast_msg_counter += 1
            # Otherwise, count single messages
            self.get_node_local_vector_msg_counter += num_asked_nodes
            # Update the counter that counts the responses
            self.node_return_local_vector_msg_counter += num_asked_nodes

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
        logging.info("Miss violations counter " + str(self.miss_violations_counter))
        logging.info("Rounds with violations counter " + str(self.rounds_with_violation_counter))
        logging.info("Total violations msg counter " + str(self.total_violations_msg_counter))
        logging.info("Node return statistics msg counter " + str(self.node_return_local_vector_msg_counter))
        logging.info("Total msgs broadcast enabled " + str(self._total_msgs_for_enabled_broadcast()) + ", and disabled " + str(self._total_msgs_for_disabled_broadcast()))
        logging.info("Num violations caused by local vector outside safe zone " + str(self.violation_origin_outside_safe_zone))
        logging.info("Num violations caused by local vector outside domain " + str(self.violation_origin_outside_domain))
        logging.info("Num violations caused by faulty safe zone " + str(self.violation_origin_faulty_safe_zone))

        logging.info("Full sync history len " + str(len(self.full_sync_history_times)))
        logging.info("Avg full sync time (ignore first time) " + str(np.mean(self.full_sync_history_times[1:])))
        logging.info("Std full sync time (ignore first time) " + str(np.std(self.full_sync_history_times[1:])))

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
                f.write("\nMiss violations counter " + str(self.miss_violations_counter))
                f.write("\nRounds with violations counter " + str(self.rounds_with_violation_counter))
                f.write("\nTotal violations msg counter " + str(self.total_violations_msg_counter))
                f.write("\nNode return statistics msg counter " + str(self.node_return_local_vector_msg_counter))
                f.write("\nTotal msgs broadcast enabled " + str(self._total_msgs_for_enabled_broadcast()) + ", and disabled " + str(self._total_msgs_for_disabled_broadcast()))
                f.write("\nNum violations caused by local vector outside safe zone " + str(self.violation_origin_outside_safe_zone))
                f.write("\nNum violations caused by local vector outside domain " + str(self.violation_origin_outside_domain))
                f.write("\nNum violations caused by faulty safe zone " + str(self.violation_origin_faulty_safe_zone))

            # Write "over time" arrays to files. Ignore the first value that is of the initial sync (and initial x0).
            file_prefix = test_folder + "/" + coordinator_name
            with open(file_prefix + "_real_function_value.csv", 'wb') as f:
                np.savetxt(f, self.real_function_value[1:])
            with open(file_prefix + "_function_approximation_error.csv", 'wb') as f:
                np.savetxt(f, self.function_approximation_error[1:])
            with open(file_prefix + "_cumulative_msgs_broadcast_enabled.csv", 'wb') as f:
                np.savetxt(f, self.cumulative_msg_for_broadcast_enabled[1:])
            with open(file_prefix + "_cumulative_msgs_broadcast_disabled.csv", 'wb') as f:
                np.savetxt(f, self.cumulative_msg_for_broadcast_disabled[1:])
            with open(file_prefix + "_cumulative_fallback_to_eager_sync.csv", 'wb') as f:
                np.savetxt(f, self.cumulative_fallback_to_eager_sync[1:])
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
                self.miss_violations_counter,
                self.rounds_with_violation_counter,
                self.total_violations_msg_counter,
                self.node_return_local_vector_msg_counter,
                self._total_msgs_for_enabled_broadcast(),
                self._total_msgs_for_disabled_broadcast()]


class CoordinatorCommon:
    
    def __init__(self, verifier, func_to_monitor, x0_len, num_nodes, error_bound=2,
                 slack_type=SlackType.Drift, sync_type=SyncType.Eager, lazy_sync_max_S=0.5):
        logging.info("CoordinatorCommon initialization: x0_len " + str(x0_len) + ", error_bound " + str(error_bound) + ", num_nodes " + str(num_nodes))
        logging.info("CoordinatorCommon initialization: slack_type " + str(slack_type) + ", sync_type " + str(sync_type) + ", lazy_sync_max_S " + str(lazy_sync_max_S))
        self.iteration = 0
        
        self.verifier = verifier  # Node that uses in lazy sync (to verify constraints) and for violation statistics.
        self.func_to_monitor = func_to_monitor
        self.x0_len = x0_len
        self.x0 = np.zeros(self.x0_len)
        self.error_bound = error_bound
        self.u_thresh = 0
        self.l_thresh = 0
        self.slack_type = slack_type
        self.sync_type = sync_type
        assert(not (slack_type == SlackType.NoSlack and SyncType.is_lazy(sync_type)))
        self.lazy_sync_max_S = lazy_sync_max_S

        self.b_faulty_safe_zone = False
        self.consecutive_neighborhood_violations_counter = 0

        # Nodes
        self.num_nodes = num_nodes
        self.nodes_x0_local = np.zeros((self.num_nodes, self.x0_len))
        # Indicates if node sent its local vector in the current iteration.
        # It could be due to violation msg from this node, or during lazy sync process.
        # It tells the coordinator, during eager sync for example, that it does not need to collect the local vector from this node.
        self.b_nodes_have_updated_local_vector = np.zeros(self.num_nodes, dtype=bool)
        self.nodes_slack = np.zeros((self.num_nodes, self.x0_len))
        self.b_nodes_have_violation = np.ones(self.num_nodes, dtype=bool)  # Trigger sync of all nodes
        self.nodes_lazy_lru_sync_counter = np.zeros(self.num_nodes)

        # Collect statistics during experiment
        self.statistics = Statistics()
    
    def set_nodes(self, nodes):
        self.nodes = nodes
        self.sync_if_needed()
    
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
    
    def _log_violation_type_and_origin(self, node_idx, violation_origin):
        # Find the type and origin of the violation and write it to log file and update statistics

        self.statistics.total_violations_msg_counter += 1

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
            # Do not assert, just log the error. This is needed in CoordinatorAutoMon, when this error can happen.
            if b_true_violation + b_false_global_violation + b_false_local_violation != 1:
                logging.info("Error: b_true_violation " + str(b_true_violation) + ", b_false_global_violation " + str(b_false_global_violation) + ", b_false_local_violation " + str(b_false_local_violation))

        self.statistics.true_violations_msg_counter += int(b_true_violation)
        self.statistics.false_global_violation_msg_counter += int(b_false_global_violation)
        self.statistics.false_local_violation_msg_counter += int(b_false_local_violation)

        if b_true_violation:
            violation_type_str = "True Violation"
        if b_false_global_violation:
            violation_type_str = "False Global Violation"
        if b_false_local_violation:
            violation_type_str = "False Local Violation"
        logging.info("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " notify " + violation_type_str)

        self.statistics.violation_origin_outside_safe_zone += int(violation_origin == ViolationOrigin.SafeZone)
        self.statistics.violation_origin_outside_domain += int(violation_origin == ViolationOrigin.Domain)
        self.statistics.violation_origin_faulty_safe_zone += int(violation_origin == ViolationOrigin.FaultySafeZone)

    def notify_violation(self, node_idx, x, violation_origin):
        # This function is called by a node to notify the coordinator about violation
        self._log_violation_type_and_origin(node_idx, violation_origin)
        self.b_nodes_have_violation[node_idx] = True
        self.nodes_x0_local[node_idx] = x
        self.b_nodes_have_updated_local_vector[node_idx] = True

        if violation_origin == ViolationOrigin.Domain:
            self.consecutive_neighborhood_violations_counter += 1
        else:
            self.consecutive_neighborhood_violations_counter = 0

        if violation_origin == ViolationOrigin.FaultySafeZone:
            # Should perform full sync to resolve the issue
            logging.info("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " notify faulty safe zone violation. Trigger full sync.")
            self.b_faulty_safe_zone = True

    def _retrieve_nodes_local_vector(self, outdated_nodes_indices):
        if len(outdated_nodes_indices) == 0:
            return
        
        logging.info("Iteration " + str(self.iteration) + ": Coordinator about to ask " + str(len(outdated_nodes_indices)) + " nodes for statistics. Nodes " + str(outdated_nodes_indices))
        self.statistics.update_node_local_vector_messages_statistics(self.iteration, len(outdated_nodes_indices))
        
        for node_idx in outdated_nodes_indices:
            assert(not self.b_nodes_have_updated_local_vector[node_idx])
            logging.info("Iteration " + str(self.iteration) + ": Coordinator asks node " + str(node_idx) + " for statistics")
            x = self.nodes[node_idx].get_local_vector()
            self.nodes_x0_local[node_idx] = x
            self.b_nodes_have_updated_local_vector[node_idx] = True
            logging.info("Iteration " + str(self.iteration) + ": Node " + str(node_idx) + " returns to coordinator with statistics")
    
    def _get_x0_and_slack_for_node_group(self, nodes_indices):
        x0 = np.zeros(self.x0_len)
        slack = np.zeros(self.x0_len)
        
        # Get stats from nodes with outdated statistics
        outdated_nodes_indices = [node_idx for node_idx in nodes_indices if not self.b_nodes_have_updated_local_vector[node_idx]]
        self._retrieve_nodes_local_vector(outdated_nodes_indices)
        
        for node_idx in nodes_indices:
            x0 += self.nodes_x0_local[node_idx]
            slack += self.nodes_slack[node_idx]
        
        return x0, slack
    
    def _eager_sync(self):
        start = timer()

        # Collect all local statistic vectors from all the nodes and compute new x0 and local constrains.
        # Set all nodes with the new x0 value and constraints
        new_x0, _ = self._get_x0_and_slack_for_node_group(list(range(self.num_nodes)))
        new_x0 = new_x0 / self.num_nodes

        # Sanity check: verify that new_x0 is the same one as the verifier x (which is the global vector)
        global_x = self.verifier.get_local_vector()
        assert(np.all(global_x - new_x0 < 1e-10))
        logging.info("Iteration " + str(self.iteration) + ": About to sync the value " + str(new_x0))
        
        self.x0 = new_x0
        # Updating the thresholds to make sure that that the new x0 is inside the safe zone.
        self._update_l_u_threshold()
        
        # Update the slacks to all nodes, and sync all nodes
        self._allocate_slacks(self.x0, list(range(self.num_nodes)))
        self._sync_nodes(list(range(self.num_nodes)), sync_type="full")
        
        # Sync also verifier. Since verifier.x equals new_x0, no slack is ever needed.
        self._sync_verifier()
        # new_x0 must be inside the safe zone. We can make sure by checking that verifier.x
        # is inside the safe zone since verifier.x equals new_x0.
        assert(self._global_vector_inside_effective_safe_zone())

        self.b_faulty_safe_zone = False

        end = timer()

        self.statistics.full_sync_history.append((self.iteration, new_x0))  # For testing: keep the iteration and the new x0
        self.statistics.full_sync_history_times.append(end - start)
        
    def _lazy_sync(self):
        b_eager_sync_fallback = False
        S_max_size = np.round(self.lazy_sync_max_S * self.num_nodes)
        S = np.nonzero(self.b_nodes_have_violation)[0]
        S_not = np.nonzero(np.logical_not(self.b_nodes_have_violation))[0]
        
        if self.sync_type == SyncType.LazyRandom:
            # Arrange S_not (the nodes without violations) in random order
            np.random.shuffle(S_not)
        if self.sync_type == SyncType.LazyLRU:
            # Arrange S_not (the nodes without violations) according to LRU
            S_not_lru_counters = self.nodes_lazy_lru_sync_counter[S_not]
            S_not = S_not[S_not_lru_counters.argsort()]
        
        logging.info("Iteration " + str(self.iteration) + ": " + str(len(S)) + " nodes with violation")
        
        # Initiate S with the nodes with violations
        S_x0, S_slack = self._get_x0_and_slack_for_node_group(S)
        
        # Add random nodes to S until the convex combination of the vectors (x_i-s_i) is in the safe zone
        next_extra_node = 0
        while len(S) < S_max_size and not self.verifier.inside_effective_safe_zone(S_x0/len(S) - S_slack/len(S)):
            node_idx = S_not[next_extra_node]
            next_extra_node += 1
            S = np.concatenate((S, [node_idx]))
            node_x, node_slack = self._get_x0_and_slack_for_node_group([node_idx])
            S_x0 += node_x
            S_slack += node_slack
        
        if len(S) > S_max_size or not self.verifier.inside_effective_safe_zone(S_x0/len(S) - S_slack/len(S)):
            logging.info("Iteration " + str(self.iteration) + ": fallback to eager sync from lazy sync !!!!!!!!!!!!!!!!!!")
            self._eager_sync()
            b_eager_sync_fallback = True
            # Reset the LRU counters of all nodes
            self.nodes_lazy_lru_sync_counter = np.zeros(self.num_nodes)
        else:
            logging.info("Iteration " + str(self.iteration) + ": used " + str(len(S)) + " nodes in lazy sync. Nodes " + str(S))
            # Allocate slack and sync nodes
            self._allocate_slacks(S_x0/len(S) - S_slack/len(S), S)
            self._sync_nodes(S, sync_type="lazy")
            # Update the LRU counters of the nodes in S
            self.nodes_lazy_lru_sync_counter[S] += 1

        return b_eager_sync_fallback

    def _check_and_report_miss_violations(self):
        # Check for miss violations (false negative). It is only possible to have miss violations in CoordinatorAutoMon
        # in case the coordinator didn't find the real min/max eigenvalue, and in CoordinatorRLV.
        # In that case there is violation of the admissible region, but no violation from any of the nodes.
        # We check it here, since this function is called after each round of set_new_data_point() for all the nodes.
        if (not np.any(self.b_nodes_have_violation)) and (not self._global_vector_inside_admissible_region()):
            self.statistics.miss_violations_counter += 1
            logging.info("Iteration " + str(self.iteration) + ": Warning: found true violation without any node violation.")
            if self.b_violation_strict:
                assert False, "Error: found true violation without any node violation when running in strict mode."

    def sync_if_needed(self):
        b_eager_sync = False
        self._check_and_report_miss_violations()

        b_violation = np.any(self.b_nodes_have_violation)

        if b_violation:
            b_eager_sync = True
            if self.b_faulty_safe_zone:
                self._eager_sync()
            elif self.coordinator_name == "AutoMon" and self.b_fix_neighborhood_dynamically and self.consecutive_neighborhood_violations_counter > self.neighborhood_violation_counter_threshold:
                self._eager_sync()
                self.consecutive_neighborhood_violations_counter = 0
            elif self.sync_type == SyncType.Eager:
                self._eager_sync()
            elif SyncType.is_lazy(self.sync_type):
                b_eager_sync = self._lazy_sync()  # Returns indication if there was a fallback to eager sync or not
            else:
                assert False, "sync_type must be one of SyncType options"
        
        self.iteration += 1
        # From now on need to get the updated node statistics, therefore, indicate
        # that the current statistics are outdated.
        self.b_nodes_have_updated_local_vector = np.zeros(self.num_nodes, dtype=bool)
        
        self.statistics.update_sync_statistics(self.func_to_monitor(self.verifier.get_local_vector()),
                                               self.func_to_monitor(self.x0), b_violation, b_eager_sync)
            
    def _allocate_slacks(self, x0, nodes_indices):
        for node_idx in nodes_indices:
            if self.slack_type == SlackType.NoSlack:
                slack = np.zeros_like(x0)
            if self.slack_type == SlackType.Drift:
                slack = self.nodes_x0_local[node_idx] - x0
            self.nodes_slack[node_idx] = slack
        
        assert(np.isclose(np.sum(self.nodes_slack), 0))
    
    def _sync_verifier(self):
        # Since verifier.x equals new_x0, no slack is ever needed.
        raise NotImplementedError("To be implemented by inherent class")
        
    def _sync_nodes(self, nodes_indices, sync_type="full"):
        logging.info("Iteration " + str(self.iteration) + ": Coordinator about to sync " + str(len(nodes_indices)) + " nodes. Nodes " + str(nodes_indices))
        self.statistics.update_sync_messages_statistics(self.iteration, len(nodes_indices))
                
        for node_idx in nodes_indices:
            logging.info("Iteration " + str(self.iteration) + ": Coordinator syncs node " + str(node_idx))
            self._sync_node(node_idx, sync_type)
            self.b_nodes_have_violation[node_idx] = False
    
    def _sync_node(self, node_idx, sync_type="full"):
        raise NotImplementedError("To be implemented by inherent class")
    
    def _update_l_u_threshold(self):
        f = self.func_to_monitor(self.x0)
        self.l_thresh = f - self.error_bound
        self.u_thresh = f + self.error_bound
        logging.info("Iteration " + str(self.iteration) + ": About to sync the thresholds " + str(self.l_thresh) + "," + str(self.u_thresh))

    def dump_stats(self, test_folder):
        self.statistics.dump_stats(test_folder, self.coordinator_name)
        return self.statistics.full_sync_history, self.statistics.get_msg_counters()
