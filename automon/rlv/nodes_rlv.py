from automon.functions_to_monitor import func_dnn_intrusion_detection, func_inner_product, func_kld, func_mlp, \
    func_quadratic_inverse, func_quadratic, func_rozenbrock
from automon.rlv.node_common_rlv import NodeCommonRLV


class NodeDnnIntrusionDetectionRLV(NodeCommonRLV):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonRLV.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_dnn_intrusion_detection)


class NodeInnerProductRLV(NodeCommonRLV):
    def __init__(self, idx=0, x0_len=2, domain=None):
        NodeCommonRLV.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_inner_product)


class NodeKLDRLV(NodeCommonRLV):
    def __init__(self, idx=0, x0_len=2, domain=None):
        NodeCommonRLV.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_kld)


class NodeMlpRLV(NodeCommonRLV):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonRLV.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_mlp)


class NodeQuadraticInverseRLV(NodeCommonRLV):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonRLV.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_quadratic_inverse)


class NodeQuadraticRLV(NodeCommonRLV):
    def __init__(self, idx=0, x0_len=1, domain=None):
        NodeCommonRLV.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_quadratic)


class NodeRozenbrockRLV(NodeCommonRLV):
    def __init__(self, idx=0, x0_len=2, domain=None):
        NodeCommonRLV.__init__(self, idx, x0_len=x0_len, domain=domain, func_to_monitor=func_rozenbrock)
