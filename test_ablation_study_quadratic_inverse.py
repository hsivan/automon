from auto_mon_monitoring.node_quadratic_inverse_auto_mon import NodeQuadraticInverseAutoMon
from coordinators.coordinator_auto_mon import CoordinatorAutoMon, DomainType
from coordinators.coordinator_rlv import CoordinatorRLV
from data_generator import DataGeneratorQuadraticInverse
from coordinators.coordinator_common import SlackType, SyncType
from functions_to_update_local_vector import update_local_vector_average
from rlv_monitoring.node_quadratic_inverse_rlv import NodeQuadraticInverseRLV
from test_figures.plot_monitoring_stats_ablation_study import plot_monitoring_stats_graph_and_barchart
from test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from stats_analysis_utils import plot_figures
import logging
from object_factory import get_objects
from functions_to_monitor import func_quadratic_inverse
import numpy as np
from test_figures.plot_quadratic_inverse_surface import draw_f_contour_and_node_trail, draw_f

if __name__ == "__main__":
    try:
        test_folder = start_test("ablation_study_quadratic_inverse")
        np.random.seed(0)

        conf = get_config(num_nodes=4, num_iterations=1020, sliding_window_size=20, k=2, error_bound=0.02,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value,
                          domain_type=DomainType.Relative.value, neighborhood_size=3)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorQuadraticInverse(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                       k=conf["k"], test_folder=test_folder)

        logging.info("\n ###################### Start quadratic inverse RLV test (no ADCD no slack) ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeQuadraticInverseRLV, CoordinatorRLV, conf, conf["k"], func_quadratic_inverse)
        coordinator.coordinator_name = "no ADCD no slack"
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        logging.info("\n ###################### Start quadratic inverse RLV test (no ADCD) ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeQuadraticInverseRLV, CoordinatorRLV, conf, conf["k"], func_quadratic_inverse)
        coordinator.coordinator_name = "no ADCD"
        coordinator.slack_type = SlackType.Drift
        coordinator.sync_type = SyncType.LazyLRU
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        logging.info("\n ###################### Start quadratic inverse AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes, verifier = get_objects(NodeQuadraticInverseAutoMon, CoordinatorAutoMon, conf, conf["k"], func_quadratic_inverse)
        run_test(data_generator, coordinator, nodes, verifier, test_folder, conf["sliding_window_size"], update_local_vector_average)

        plot_figures(test_folder)
        draw_f_contour_and_node_trail(data_generator.data, test_folder)
        draw_f(test_folder)
        plot_monitoring_stats_graph_and_barchart(test_folder, "quadratic_inverse", test_folder + "/")

    finally:
        end_test()
