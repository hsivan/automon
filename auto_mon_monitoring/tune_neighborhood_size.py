from coordinators.coordinator_auto_mon import CoordinatorAutoMon
from test_utils import run_test
import logging
import numpy as np
from object_factory import get_objects
import traceback


class ViolationsStats:
    def __init__(self, neighborhood_size, total_violations, safe_zone_violations, neighborhood_violations):
        self.neighborhood_size = neighborhood_size
        self.total_violations = total_violations
        self.safe_zone_violations = safe_zone_violations
        self.neighborhood_violations = neighborhood_violations

    def __repr__(self):
        return '(' + str(self.neighborhood_size) + ', ' + str(self.total_violations) + ', ' + str(self.safe_zone_violations) + ', ' + str(self.neighborhood_violations) + ')'


def _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, neighborhood_size, sliding_window_size, func_update_local_vector, b_check_violation_every_sample):
    conf['neighborhood_size'] = neighborhood_size
    logging.info("Checking neighborhood size " + str(neighborhood_size))
    data_generator.reset()
    coordinator, nodes, verifier = get_objects(NodeClass, CoordinatorAutoMon, conf, x0_len, func_monitor)
    if coordinator.b_hessian_const:
        # Using ADCD-E, the neighborhood should be the entire domain. No need for tuning.
        logging.info("The Hessian is constant, no need for neighborhood tuning")
        return None
    coordinator.b_tune_neighborhood_mode = True
    coordinator.b_fix_neighborhood_dynamically = False  # Should not change neighborhood size dynamically during tuning
    run_test(data_generator, coordinator, nodes, verifier, None, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
    violations_stats = ViolationsStats(neighborhood_size, coordinator.statistics.total_violations_msg_counter,
                                       coordinator.statistics.violation_origin_outside_safe_zone, coordinator.statistics.violation_origin_outside_domain)
    return violations_stats


def tune_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, sliding_window_size, func_update_local_vector, b_check_violation_every_sample=False):
    data_generator.set_neighborhood_tuning_state()
    if data_generator.get_num_iterations() == 0:
        logging.info("Error: called tune_neighborhood_size() with num_iterations_for_tuning=0")
        raise Exception

    logging.info("\n ###################### Start tuning neighborhood size ######################")
    if conf["domain"] is None:
        entire_domain_size = np.inf
    else:
        entire_domain_size = conf["domain"][1] - conf["domain"][0]

    try:
        # Run first time with initial neighborhood size 1.0 to verify if the Hessian is constant (in that case there is no need for tuning)
        violations_stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, 1.0, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
        if violations_stats is None:
            data_generator.set_monitoring_state()
            return None

        # Start by finding neighborhood size that obtains at least some neighborhood violations
        while violations_stats.neighborhood_violations == 0:
            violations_stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, violations_stats.neighborhood_size / 2, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
        violations_stats_arr = [violations_stats]

        # Find small enough neighborhood size that obtains 0 safe zone violations
        while violations_stats.safe_zone_violations > 0 and violations_stats.neighborhood_violations < 2 * violations_stats.safe_zone_violations:
            violations_stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, violations_stats.neighborhood_size / 2, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
            violations_stats_arr.insert(0, violations_stats)  # Keep sorted by neighborhood size, from smallest to largest

        # violations_stats of the largest neighborhood size
        violations_stats = violations_stats_arr[-1]

        # Find large enough neighborhood size that obtains 0 neighborhood violations
        while violations_stats.neighborhood_violations > 0 and 2 * violations_stats.neighborhood_size <= entire_domain_size:
            violations_stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, violations_stats.neighborhood_size * 2, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
            violations_stats_arr.append(violations_stats)  # Keep sorted by neighborhood size, from smallest to largest

        # At this point we have range for neighborhood size. Divide the range to 10 and test each neighborhood size.
        smallest_neighborhood_size, largest_neighborhood_size = violations_stats_arr[0].neighborhood_size, violations_stats_arr[-1].neighborhood_size
        logging.info("Found range for neighborhood size. Min: " + str(smallest_neighborhood_size) + ", Max: " + str(largest_neighborhood_size))
        if smallest_neighborhood_size != largest_neighborhood_size:
            neighborhood_sizes = np.linspace(smallest_neighborhood_size, largest_neighborhood_size, 10)[1:-1]
            for neighborhood_size in neighborhood_sizes:
                violations_stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, neighborhood_size, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
                violations_stats_arr.append(violations_stats)
            violations_stats_arr = sorted(violations_stats_arr, key=lambda stats: stats.neighborhood_size)  # Sort by neighborhood size, from smallest to largest

    except Exception as e:
        logging.info("Exception: " + str(e))
        logging.info(traceback.print_exc())
        raise

    data_generator.set_monitoring_state()
    logging.info("violations_stats_arr: " + str(violations_stats_arr))

    optimal_violations_stats = min(violations_stats_arr,  key=lambda stats: stats.total_violations)  # Find violations_stats with the minimal total num violations
    logging.info("The optimal neighborhood size for error bound " + str(conf["error_bound"]) + " is: " + str(optimal_violations_stats.neighborhood_size))
    logging.info("\n ###################### Finished tuning neighborhood size ######################")
    return optimal_violations_stats.neighborhood_size
