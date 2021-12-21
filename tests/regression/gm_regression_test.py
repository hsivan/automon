import numpy as np
from automon.coordinator_common import SlackType, SyncType
from automon.gm.coordinator_gm import CoordinatorGM
from automon_utils.data_generator import DataGeneratorEntropy, DataGeneratorVariance
from automon.gm.node_entropy_gm import NodeEntropyGM
from automon.gm.node_variance_gm import NodeVarianceGM
from automon_utils.functions_to_monitor import func_entropy, func_variance
from automon_utils.test_utils import start_test, end_test, run_test, get_config
from automon_utils.object_factory import get_objects
from tests.regression.regression_test_utils import compare_results

regression_test_files_folder = "./regression_test_files_gm/"
    

if __name__ == "__main__":
    try:
        test_folder = start_test("gm_regression")

        print("\nRun entropy test with Drift Slack and Eager Sync")
        conf = get_config(num_nodes=2, num_iterations=30, sliding_window_size=20, d=3, error_bound=0.2,
                          slack_type=SlackType.Drift.value, sync_type=SyncType.Eager.value)
        data_generator = DataGeneratorEntropy(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name=regression_test_files_folder + "data_file_entropy.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        coordinator, nodes = get_objects(NodeEntropyGM, CoordinatorGM, conf, func_entropy)
        sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder)
        compare_results(regression_test_files_folder, sync_history, msg_counters, "entropy", "drift", "eager")
        
        print("\nRun entropy test with Drift Slack and Lazy Random Sync")
        np.random.seed(seed=1)
        conf = get_config(num_nodes=5, num_iterations=12, sliding_window_size=5, d=3, error_bound=0.2, slack_type=SlackType.Drift.value, sync_type=SyncType.LazyRandom.value)
        data_generator = DataGeneratorEntropy(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name=regression_test_files_folder + "data_file_entropy.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        coordinator, nodes = get_objects(NodeEntropyGM, CoordinatorGM, conf, func_entropy)
        sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder)
        compare_results(regression_test_files_folder, sync_history, msg_counters, "entropy", "drift", "lazy_random")
        
        print("\nRun entropy test with Drift Slack and Lazy LRU Sync")
        np.random.seed(seed=1)
        conf = get_config(num_nodes=5, num_iterations=12, sliding_window_size=5, d=3, error_bound=0.2, slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        data_generator = DataGeneratorEntropy(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name=regression_test_files_folder + "data_file_entropy.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        coordinator, nodes = get_objects(NodeEntropyGM, CoordinatorGM, conf, func_entropy)
        sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder)
        compare_results(regression_test_files_folder, sync_history, msg_counters, "entropy", "drift", "lazy_lru")
        
        print("\nRun entropy test with No Slack and Lazy Random Sync")
        try:
            b_assertion = False
            conf = get_config(num_nodes=2, num_iterations=30, sliding_window_size=20, d=3, error_bound=0.2,
                              slack_type=SlackType.NoSlack.value, sync_type=SyncType.LazyRandom.value)
            coordinator, nodes = get_objects(NodeEntropyGM, CoordinatorGM, conf, func_entropy)
        except AssertionError:
            print("This combination should throw exception. No Slack must come with Eager Sync.")
            b_assertion = True
        assert b_assertion
        
        print("\nRun entropy test with No Slack and Eager Sync")
        conf = get_config(num_nodes=2, num_iterations=30, sliding_window_size=20, d=3, error_bound=0.2, slack_type=SlackType.NoSlack.value, sync_type=SyncType.Eager.value)
        data_generator = DataGeneratorEntropy(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name=regression_test_files_folder + "data_file_entropy.txt", d=conf["d"], sliding_window_size=conf["sliding_window_size"])
        data_generator.reset()
        coordinator, nodes = get_objects(NodeEntropyGM, CoordinatorGM, conf, func_entropy)
        sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder)
        compare_results(regression_test_files_folder, sync_history, msg_counters, "entropy", "no", "eager")
        


        print("\nRun variance test with Drift Slack and Eager Sync")
        conf = get_config(num_nodes=2, num_iterations=100, sliding_window_size=5, d=2, error_bound=2, slack_type=SlackType.Drift.value, sync_type=SyncType.Eager.value)
        data_generator = DataGeneratorVariance(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name=regression_test_files_folder+"data_file_variance.txt", sliding_window_size=conf["sliding_window_size"])
        coordinator, nodes = get_objects(NodeVarianceGM, CoordinatorGM, conf, func_variance)
        sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder)
        compare_results(regression_test_files_folder, sync_history, msg_counters, "variance", "drift", "eager")
        
        print("\nRun variance test with Drift Slack and Lazy Random Sync")
        np.random.seed(seed=1)
        conf = get_config(num_nodes=5, num_iterations=40, sliding_window_size=5, d=2, error_bound=2, slack_type=SlackType.Drift.value, sync_type=SyncType.LazyRandom.value)
        data_generator = DataGeneratorVariance(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name=regression_test_files_folder + "data_file_variance.txt", sliding_window_size=conf["sliding_window_size"])
        coordinator, nodes = get_objects(NodeVarianceGM, CoordinatorGM, conf, func_variance)
        sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder)
        compare_results(regression_test_files_folder, sync_history, msg_counters, "variance", "drift", "lazy_random")
        
        print("\nRun variance test with Drift Slack and Lazy LRU Sync")
        np.random.seed(seed=1)
        conf = get_config(num_nodes=5, num_iterations=40, sliding_window_size=3, d=2, error_bound=2, slack_type=SlackType.Drift.value, sync_type=SyncType.LazyLRU.value)
        data_generator = DataGeneratorVariance(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name=regression_test_files_folder+"data_file_variance.txt", sliding_window_size=conf["sliding_window_size"])
        coordinator, nodes = get_objects(NodeVarianceGM, CoordinatorGM, conf, func_variance)
        sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder)
        compare_results(regression_test_files_folder, sync_history, msg_counters, "variance", "drift", "lazy_lru")
        
        print("\nRun variance test with No Slack and Lazy Random Sync")
        try:
            b_assertion = False
            conf = get_config(num_nodes=2, num_iterations=100, sliding_window_size=5, d=2, error_bound=2,
                              slack_type=SlackType.NoSlack.value, sync_type=SyncType.LazyRandom.value)
            coordinator, nodes = get_objects(NodeVarianceGM, CoordinatorGM, conf, func_variance)
        except AssertionError:
            print("This combination should throw exception. No Slack must come with Eager Sync.")
            b_assertion = True
        assert b_assertion

        print("\nRun variance test with No Slack and Eager Sync")
        conf = get_config(num_nodes=2, num_iterations=100, sliding_window_size=5, d=2, error_bound=2, slack_type=SlackType.NoSlack.value, sync_type=SyncType.Eager.value)
        data_generator = DataGeneratorVariance(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"], data_file_name=regression_test_files_folder+"data_file_variance.txt", sliding_window_size=conf["sliding_window_size"])
        coordinator, nodes = get_objects(NodeVarianceGM, CoordinatorGM, conf, func_variance)
        sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder)
        compare_results(regression_test_files_folder, sync_history, msg_counters, "variance", "no", "eager")
    
    finally:
        end_test()
