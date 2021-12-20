import numpy as np
from automon.automon.node_common_automon import NodeCommonAutoMon
from utils.functions_to_monitor import func_cosine_similarity, func_dnn_intrusion_detection, func_entropy, func_inner_product, \
    func_kld, func_mlp, func_quadratic, func_quadratic_inverse, func_rozenbrock, func_sine, func_variance


class NodeDnnIntrusionDetectionAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_dnn_intrusion_detection)


class NodeInnerProductAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=2, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_inner_product)


class NodeKLDAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=2, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_kld)


class NodeMlpAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_mlp)


class NodeQuadraticAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_quadratic)


class NodeQuadraticInverseAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_quadratic_inverse)


class NodeRozenbrockAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=2, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_rozenbrock)


class NodeSineAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_sine)


class NodeVarianceAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=2, domain=None):
        assert (x0_len == 2)  # The local vector is the first and second momentum
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, min_f_val=0.0, domain=domain, func_to_monitor=func_variance)


class NodeEntropyAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, max_f_val=func_entropy(np.ones(x0_len, dtype=np.float) / x0_len), min_f_val=0.0, domain=domain, func_to_monitor=func_entropy)


class NodeCosineSimilarityAutoMon(NodeCommonAutoMon):
    def __init__(self, idx=0, x0_len=2, domain=None):
        NodeCommonAutoMon.__init__(self, idx, x0_len=x0_len, max_f_val=1.0, min_f_val=-1.0, domain=domain, func_to_monitor=func_cosine_similarity)
