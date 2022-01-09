from automon import AutomonNode, AutomonCoordinator, CbCosineSimilarityNode, CbCoordinator, SlackType, SyncType
from test_utils.functions_to_monitor import func_cosine_similarity
from test_utils.data_generator import DataGeneratorCosineSimilarity
from test_utils.object_factory import get_objects
from test_utils.test_utils import start_test, end_test, run_test, get_config, write_config_to_file
from test_utils.stats_analysis_utils import plot_monitoring_stats
import logging


if __name__ == "__main__":
    try:
        test_folder = start_test("compare_methods_cosine_similarity")
        
        conf = get_config(num_nodes=10, num_iterations=50, sliding_window_size=5, d=10, error_bound=0.5,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        write_config_to_file(test_folder, conf)

        data_generator = DataGeneratorCosineSimilarity(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], d=conf["d"], test_folder=test_folder, sliding_window_size=conf["sliding_window_size"])
        
        logging.info("\n###################### Start cosine similarity CB test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(CbCosineSimilarityNode, CbCoordinator, conf, func_cosine_similarity)
        run_test(data_generator, coordinator, nodes, test_folder)
        
        logging.info("\n###################### Start cosine similarity AutoMon test ######################")
        data_generator.reset()
        coordinator, nodes = get_objects(AutomonNode, AutomonCoordinator, conf, func_cosine_similarity, max_f_val=1.0, min_f_val=-1.0)
        run_test(data_generator, coordinator, nodes, test_folder)
        
        plot_monitoring_stats(test_folder)
    
    finally:
        end_test()
