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
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the DARTS specific MixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", DARTSTopKMixedOp(primitives))

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

    def __init__(self, primitives, top_k=1):
        super().__init__(primitives)
        self.top_k = min(top_k, len(primitives))

    def process_weights(self, weights):
        mask = torch.zeros_like(weights)
        mask[torch.min(torch.topk(weights, self.top_k).values) <= weights] = 1
        return torch.softmax(weights * mask, dim=-1)

    def apply_weights(self, x, weights):
        return sum([weights[idx] * self.primitives[idx](x, None) for idx in torch.topk(weights, self.top_k).indices])