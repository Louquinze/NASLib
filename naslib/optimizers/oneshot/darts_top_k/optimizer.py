import numpy as np
import torch
import logging
from torch.autograd import Variable

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSMixedOp, DARTSOptimizer
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.utils.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

import naslib.search_spaces.core.primitives as ops

logger = logging.getLogger(__name__)


class DARTSTopKOptimizer(DARTSOptimizer):
    """
    Implementation of the DARTS paper as in
        Liu et al. 2019: DARTS: Differentiable Architecture Search.
    """
    @staticmethod
    def update_ops(edge, top_k=None):
        """
        Function to replace the primitive ops at the edges
        with the DARTS specific MixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", DARTSTopKMixedOp(primitives, top_k=top_k))

    def __init__(
        self,
        config,
        op_optimizer=torch.optim.SGD,
        arch_optimizer=torch.optim.Adam,
        loss_criteria=torch.nn.CrossEntropyLoss(),
        top_k=1
    ):
        """
        Initialize a new instance.

        Args:

        """
        super(DARTSTopKOptimizer, self).__init__(config, op_optimizer, arch_optimizer, loss_criteria)


class DARTSTopKMixedOp(DARTSMixedOp):
    """
    Continous relaxation of the discrete search space.
    """

    def __init__(self, primitives, top_k):
        super().__init__(primitives)
        assert top_k != None
        self.top_k = min(top_k, len(primitives))

    def process_weights(self, weights):
        topk = torch.topk(weights, self.top_k)
        min_threshold = torch.min(topk.values)
        mask = torch.zeros_like(weights)
        mask[weights >= min_threshold] = 1 / (sum(topk.values) + 1e-05)
        return weights * mask

    def apply_weights(self, x, weights):
        res = 0
        for w, op in zip(weights, self.primitives):
            if w > 0:
                res += w * op(x, None)
        return res