import numpy as np
from automon.coordinator_common import SlackType, SyncType
from automon.automon.coordinator_automon import CoordinatorAutoMon


def get_node(NodeClass, domain, x0_len, node_idx, func_to_monitor, max_f_val=np.inf, min_f_val=-np.inf):
    if max_f_val != np.inf or min_f_val != -np.inf:
        node = NodeClass(idx=node_idx, x0_len=x0_len, domain=domain, func_to_monitor=func_to_monitor, max_f_val=max_f_val, min_f_val=min_f_val)
    else:
        node = NodeClass(idx=node_idx, x0_len=x0_len, domain=domain, func_to_monitor=func_to_monitor)
    return node


def get_nodes(NodeClass, domain, x0_len, num_nodes, func_to_monitor, max_f_val=np.inf, min_f_val=-np.inf):
    nodes = [get_node(NodeClass, domain, x0_len, node_idx, func_to_monitor, max_f_val, min_f_val) for node_idx in range(num_nodes)]
    return nodes


def get_coordinator(CoordinatorClass, NodeClass, conf, func_to_monitor, max_f_val=np.inf, min_f_val=-np.inf):
    # The verifier is just another node that is held by the coordinator, and its local vector is based on window size
    # that is the sum of all other nodes window sizes.
    # x0 is initialized from the nodes after all the sliding windows are full, in the first eager sync.
    verifier = get_node(NodeClass, conf["domain"], conf["d"], -1, func_to_monitor, max_f_val, min_f_val)

    if CoordinatorClass is CoordinatorAutoMon:
        # AutoMon coordinator have extra parameter neighborhood_size
        coordinator = CoordinatorClass(verifier, conf["num_nodes"],
                                       slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                       error_bound=conf["error_bound"], domain=conf["domain"],
                                       neighborhood_size=conf["neighborhood_size"])
    else:
        coordinator = CoordinatorClass(verifier, conf["num_nodes"],
                                       slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                       error_bound=conf["error_bound"], domain=conf["domain"])
    return coordinator


def get_objects(NodeClass, CoordinatorClass, conf, func_to_monitor, max_f_val=np.inf, min_f_val=-np.inf):
    nodes = get_nodes(NodeClass, conf["domain"], conf["d"], conf["num_nodes"], func_to_monitor, max_f_val, min_f_val)
    coordinator = get_coordinator(CoordinatorClass, NodeClass, conf, func_to_monitor, max_f_val, min_f_val)
    return coordinator, nodes
