import numpy as np
from automon import SlackType, SyncType, AutomonCoordinator, RlvCoordinator


def _get_node(NodeClass, domain, d, node_idx, func_to_monitor, max_f_val=np.inf, min_f_val=-np.inf):
    if max_f_val != np.inf or min_f_val != -np.inf:
        node = NodeClass(idx=node_idx, d=d, domain=domain, func_to_monitor=func_to_monitor, max_f_val=max_f_val, min_f_val=min_f_val)
    else:
        node = NodeClass(idx=node_idx, d=d, domain=domain, func_to_monitor=func_to_monitor)
    return node


def _get_coordinator(CoordinatorClass, NodeClass, conf, func_to_monitor, max_f_val=np.inf, min_f_val=-np.inf):
    if CoordinatorClass is AutomonCoordinator:
        coordinator = AutomonCoordinator(conf["num_nodes"], func_to_monitor, slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                         error_bound=conf["error_bound"], neighborhood_size=conf["neighborhood_size"], d=conf["d"],
                                         max_f_val=max_f_val, min_f_val=min_f_val, domain=conf["domain"])
    elif CoordinatorClass is RlvCoordinator:
        coordinator = RlvCoordinator(conf["num_nodes"], func_to_monitor, slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                     error_bound=conf["error_bound"], d=conf["d"], max_f_val=max_f_val, min_f_val=min_f_val, domain=conf["domain"])
    else:
        coordinator = CoordinatorClass(NodeClass, conf["num_nodes"], func_to_monitor, slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                       error_bound=conf["error_bound"], d=conf["d"], domain=conf["domain"])
    return coordinator


def get_objects(NodeClass, CoordinatorClass, conf, func_to_monitor, max_f_val=np.inf, min_f_val=-np.inf):
    nodes = [_get_node(NodeClass, conf["domain"], conf["d"], node_idx, func_to_monitor, max_f_val, min_f_val) for node_idx in range(conf["num_nodes"])]
    coordinator = _get_coordinator(CoordinatorClass, NodeClass, conf, func_to_monitor, max_f_val, min_f_val)
    return coordinator, nodes
