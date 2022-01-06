from automon.auto_mon.node_automon import NodeAutoMon
from automon.rlv.node_rlv import NodeRLV
from test_utils.functions_to_monitor import func_quadratic_inverse
from automon.auto_mon.coordinator_automon import CoordinatorAutoMon
from automon.rlv.coordinator_rlv import CoordinatorRLV
from test_utils.data_generator import DataGeneratorQuadraticInverse
from automon.coordinator_common import SlackType, SyncType
from experiments.visualization.plot_monitoring_stats_ablation_study import plot_monitoring_stats_graph_and_barchart
from test_utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging
from test_utils.object_factory import get_objects
import numpy as np
from experiments.visualization.plot_quadratic_inverse_surface import draw_f_contour_and_node_trail, draw_f
logging = logging.getLogger('automon')

if __name__ == "__main__":
    try:
        test_folder = start_test("ablation_study_quadratic_inverse")
        np.random.seed(0)

        conf = get_config(num_nodes=4, num_iterations=1020, sliding_window_size=20, d=2, error_bound=0.02,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value, neighborhood_size=3)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorQuadraticInverse(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                                                       d=conf["d"], test_folder=test_folder, sliding_window_size=conf["sliding_window_size"])

        logging.info("\n###################### Start quadratic inverse RLV test (no ADCD no slack) ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeRLV, CoordinatorRLV, conf, func_quadratic_inverse)
        coordinator.coordinator_name = "no ADCD no slack"
        run_test(data_generator, coordinator, nodes, test_folder)

        logging.info("\n###################### Start quadratic inverse RLV test (no ADCD) ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeRLV, CoordinatorRLV, conf, func_quadratic_inverse)
        coordinator.coordinator_name = "no ADCD"
        coordinator.slack_type = SlackType.Drift
        coordinator.sync_type = SyncType.LazyLRU
        run_test(data_generator, coordinator, nodes, test_folder)

        logging.info("\n###################### Start quadratic inverse AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(NodeAutoMon, CoordinatorAutoMon, conf, func_quadratic_inverse)
        run_test(data_generator, coordinator, nodes, test_folder)

        plot_monitoring_stats(test_folder)
        draw_f_contour_and_node_trail(data_generator.data, test_folder)
        draw_f(test_folder)
        plot_monitoring_stats_graph_and_barchart(test_folder, "quadratic_inverse", test_folder + "/")

    finally:
        end_test()
