import numpy as np
import pickle
from automon.object_factory import get_objects
from automon.coordinator_common import SyncType, SlackType
from automon.test_utils import run_test


def compare_sync_history_lists(sync_history_1, sync_history_2):
    assert(len(sync_history_1) == len(sync_history_2))
    for sync_idx in range(len(sync_history_1)):
        assert(sync_history_1[sync_idx][0] == sync_history_2[sync_idx][0])
        assert(np.allclose(sync_history_1[sync_idx][1], sync_history_2[sync_idx][1]))


def compare_msg_counters_lists(msg_counters_1, msg_counters_2):
    assert(msg_counters_1 == msg_counters_2)


def compare_results(regression_test_files_folder, sync_history, msg_counters, type_str, slack_str, sync_str):
    expected_sync_history_str = regression_test_files_folder + "sync_history_" + type_str + "_" + slack_str + "_slack_" + sync_str + "_sync.pkl"
    expected_msg_counters_str = regression_test_files_folder + "msg_counters_" + type_str + "_" + slack_str + "_slack_" + sync_str + "_sync.pkl"
    expected_sync_history = pickle.load(open(expected_sync_history_str, 'rb'))
    expected_msg_counters = pickle.load(open(expected_msg_counters_str, 'rb'))
    compare_sync_history_lists(sync_history, expected_sync_history)
    compare_msg_counters_lists(msg_counters, expected_msg_counters)
    # pickle.dump(sync_history, open(expected_sync_history_str, 'wb'))
    # pickle.dump(msg_counters, open(expected_msg_counters_str, 'wb'))


def test_func_slack_sync_variations(coordinator_class, func_name, NodeClass, data_generator, conf, test_folder, regression_test_files_folder):
    print("\nRun " + func_name + " test with Drift Slack and Eager Sync")
    np.random.seed(seed=1)
    data_generator.reset()
    conf["slack_type"], conf["sync_type"] = SlackType.Drift.value, SyncType.Eager.value
    coordinator, nodes = get_objects(NodeClass, coordinator_class, conf)
    sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])
    compare_results(regression_test_files_folder, sync_history, msg_counters, func_name, "drift", "eager")

    print("\nRun " + func_name + " test with Drift Slack and Lazy Random Sync")
    np.random.seed(seed=1)
    data_generator.reset()
    conf["slack_type"], conf["sync_type"] = SlackType.Drift.value, SyncType.LazyRandom.value
    coordinator, nodes = get_objects(NodeClass, coordinator_class, conf)
    sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])
    compare_results(regression_test_files_folder, sync_history, msg_counters, func_name, "drift", "lazy_random")

    print("\nRun " + func_name + " test with Drift Slack and Lazy LRU Sync")
    np.random.seed(seed=1)
    data_generator.reset()
    conf["slack_type"], conf["sync_type"] = SlackType.Drift.value, SyncType.LazyLRU.value
    coordinator, nodes = get_objects(NodeClass, coordinator_class, conf)
    sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])
    compare_results(regression_test_files_folder, sync_history, msg_counters, func_name, "drift", "lazy_lru")

    print("\nRun " + func_name + " test with No Slack and Lazy Random Sync")
    try:
        b_assertion = False
        conf["slack_type"], conf["sync_type"] = SlackType.NoSlack.value, SyncType.LazyRandom.value
        _, _ = get_objects(NodeClass, coordinator_class, conf)
    except AssertionError:
        print("This combination should throw exception. No Slack must come with Eager Sync.")
        b_assertion = True
    assert b_assertion

    print("\nRun " + func_name + " test with No Slack and Eager Sync")
    np.random.seed(seed=1)
    data_generator.reset()
    conf["slack_type"], conf["sync_type"] = SlackType.NoSlack.value, SyncType.Eager.value
    coordinator, nodes = get_objects(NodeClass, coordinator_class, conf)
    sync_history, msg_counters = run_test(data_generator, coordinator, nodes, test_folder, conf["sliding_window_size"])
    compare_results(regression_test_files_folder, sync_history, msg_counters, func_name, "no", "eager")
