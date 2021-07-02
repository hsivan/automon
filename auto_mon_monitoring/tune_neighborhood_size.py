from coordinators.coordinator_auto_mon import CoordinatorAutoMon
from test_utils import run_test
import logging
import numpy as np
from object_factory import get_objects
import traceback


def _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, neighborhood_size, sliding_window_size, func_update_local_vector, b_check_violation_every_sample):
    conf['neighborhood_size'] = neighborhood_size
    logging.info("Checking neighborhood size " + str(neighborhood_size))
    data_generator.reset_for_tuning()
    coordinator, nodes, verifier = get_objects(NodeClass, CoordinatorAutoMon, conf, x0_len, func_monitor)
    if coordinator.b_hessian_const:
        # The neighborhood should be the entire domain (done in the coordinator). No need for tuning.
        logging.info("The Hessian is constant, no need for neighborhood tuning")
        return None
    coordinator.b_tune_neighborhood_mode = True
    coordinator.b_fix_neighborhood_dynamically = False  # Should not change neighborhood size dynamically during tuning
    run_test(data_generator, coordinator, nodes, verifier, None, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
    total_violations = coordinator.statistics.total_violations_msg_counter
    safe_zone_violations = coordinator.statistics.violation_origin_outside_safe_zone
    neighborhood_violations = coordinator.statistics.violation_origin_outside_domain
    return neighborhood_size, total_violations, safe_zone_violations, neighborhood_violations


def tune_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, sliding_window_size, func_update_local_vector, b_check_violation_every_sample=False):
    if data_generator.num_iterations_for_tuning == 0:
        logging.info("Error: called tune_neighborhood_size() with num_iterations_for_tuning=0")
        raise Exception

    neighborhood_size = 1.0
    original_num_iterations = data_generator.num_iterations
    data_generator.num_iterations = data_generator.num_iterations_for_tuning  # Use only first num_iterations (after window is full) for the tuning

    logging.info("\n ###################### Start tuning neighborhood size ######################")
    if conf["domain"] is None:
        entire_domain_size = np.inf
    else:
        entire_domain_size = conf["domain"][1] - conf["domain"][0]

    try:
        # Run first time with initial neighborhood size to verify if the Hessian is constant (in that case there is no need for tuning)
        stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, neighborhood_size, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
        if stats is None:
            data_generator.num_iterations = original_num_iterations  # Restore num iterations
            return None

        # Start by finding neighborhood size that obtains at least some neighborhood violations
        while stats[3] == 0:  # neighborhood_violations == 0
            neighborhood_size = neighborhood_size / 2
            stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, neighborhood_size, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
        neighborhood_sizes_arr = [stats]

        # Find small enough neighborhood size that obtains 0 safe zone violations
        stats = neighborhood_sizes_arr[0]  # stats of the smallest neighborhood size
        while stats[2] > 0 and stats[3] < 2 * stats[2]:  # The number of safe zone violations of the smallest neighborhood size
            smallest_neighborhood_size = stats[0] / 2
            stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, smallest_neighborhood_size, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
            neighborhood_sizes_arr.insert(0, stats)  # Keep sorted by neighborhood size, from smallest to largest

        # Find large enough neighborhood size that obtains 0 neighborhood violations
        stats = neighborhood_sizes_arr[-1]  # stats of the largest neighborhood size
        while stats[3] > 0 and 2 * stats[0] <= entire_domain_size:  # The number of neighborhood violations of the largest neighborhood size
            largest_neighborhood_size = stats[0] * 2
            stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, largest_neighborhood_size, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
            neighborhood_sizes_arr.append(stats)  # Keep sorted by neighborhood size, from smallest to largest

        # At this point we have range for neighborhood size. Divide the range to 10 and test each neighborhood size.
        logging.info("Found range for neighborhood size. Min: " + str(neighborhood_sizes_arr[0][0]) + ", Max: " + str(neighborhood_sizes_arr[-1][0]))
        if neighborhood_sizes_arr[0][0] != neighborhood_sizes_arr[-1][0]:
            neighborhood_sizes = np.linspace(neighborhood_sizes_arr[0][0], neighborhood_sizes_arr[-1][0], 10)
            neighborhood_sizes = neighborhood_sizes[1:-1]
            for neighborhood_size in neighborhood_sizes:
                stats = _test_neighborhood_size(func_monitor, NodeClass, conf, x0_len, data_generator, neighborhood_size, sliding_window_size, func_update_local_vector, b_check_violation_every_sample)
                neighborhood_sizes_arr.append(stats)  # Keep sorted by neighborhood size, from smallest to largest
            neighborhood_sizes_arr.sort()  # Sort by neighborhood size, from smallest to largest

    except Exception as e:
        logging.info("Exception: " + str(e))
        logging.info(traceback.print_exc())
        return

    data_generator.num_iterations = original_num_iterations  # Restore num iterations
    logging.info("neighborhood_sizes_arr: " + str(neighborhood_sizes_arr))

    optimal_stats = min(neighborhood_sizes_arr,  key=lambda x: x[1])  # Find stats with the minimal total num violations
    logging.info("The optimal neighborhood size for error bound " + str(conf["error_bound"]) + " is: " + str(optimal_stats[0]))
    logging.info("\n ###################### Finished tuning neighborhood size ######################")
    return optimal_stats[0]
