from coordinators.coordinator_common import SlackType, SyncType
from coordinators.coordinator_auto_mon import CoordinatorAutoMon, DomainType
import logging


def get_verifier_and_nodes(NodeClass, conf, x0_len):
    logging.info("sliding_window_size " + str(conf["sliding_window_size"]))

    # The verifier is just another node, but without coordinator and with window size
    # that is the sum of all other nodes window sizes
    verifier = NodeClass(idx=-1, local_vec_len=x0_len)

    nodes = []
    for idx in range(conf["num_nodes"]):
        node = NodeClass(idx=idx, local_vec_len=x0_len)
        nodes.append(node)

    return verifier, nodes


def get_coordinator(CoordinatorClass, verifier, x0_len, conf, func_to_monitor):
    if CoordinatorClass is CoordinatorAutoMon:
        # AutoMon coordinator have extra parameter domain_type and neighborhood_size
        coordinator = CoordinatorClass(verifier, func_to_monitor, x0_len, conf["num_nodes"],
                                       slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                       error_bound=conf["error_bound"], domain=conf["domain"],
                                       domain_type=DomainType(conf["domain_type"]), neighborhood_size=conf["neighborhood_size"])
    else:
        coordinator = CoordinatorClass(verifier, func_to_monitor, x0_len, conf["num_nodes"],
                                       slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                       error_bound=conf["error_bound"], domain=conf["domain"])
    return coordinator


def get_objects(NodeClass, CoordinatorClass, conf, x0_len, func_to_monitor):
    verifier, nodes = get_verifier_and_nodes(NodeClass, conf, x0_len)
    coordinator = get_coordinator(CoordinatorClass, verifier, x0_len, conf, func_to_monitor)
    return coordinator, nodes, verifier
