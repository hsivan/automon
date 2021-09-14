from automon.coordinator_common import SlackType, SyncType
from automon.automon.coordinator_automon import CoordinatorAutoMon, DomainType


def get_node(NodeClass, domain, x0_len, node_idx):
    node = NodeClass(idx=node_idx, x0_len=x0_len, domain=domain)
    return node


def get_nodes(NodeClass, domain, x0_len, num_nodes):
    nodes = []
    for idx in range(num_nodes):
        node = NodeClass(idx=idx, x0_len=x0_len, domain=domain)
        nodes.append(node)
    return nodes


def get_coordinator(CoordinatorClass, NodeClass, conf):
    # The verifier is just another node that is held by the coordinator, and its local vector is based on window size
    # that is the sum of all other nodes window sizes.
    # x0 is initialized from the nodes after all the sliding windows are full, in the first eager sync.
    verifier = NodeClass(idx=-1, x0_len=conf["d"], domain=conf["domain"])

    if CoordinatorClass is CoordinatorAutoMon:
        # AutoMon coordinator have extra parameter domain_type and neighborhood_size
        coordinator = CoordinatorClass(verifier, verifier.func_to_monitor, conf["num_nodes"],
                                       slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                       error_bound=conf["error_bound"], domain=conf["domain"],
                                       domain_type=DomainType(conf["domain_type"]), neighborhood_size=conf["neighborhood_size"])
    else:
        coordinator = CoordinatorClass(verifier, verifier.func_to_monitor, conf["num_nodes"],
                                       slack_type=SlackType(conf["slack_type"]), sync_type=SyncType(conf["sync_type"]),
                                       error_bound=conf["error_bound"], domain=conf["domain"])
    return coordinator


def get_objects(NodeClass, CoordinatorClass, conf):
    nodes = get_nodes(NodeClass, conf["domain"], conf["d"], conf["num_nodes"])
    coordinator = get_coordinator(CoordinatorClass, NodeClass, conf)
    return coordinator, nodes
