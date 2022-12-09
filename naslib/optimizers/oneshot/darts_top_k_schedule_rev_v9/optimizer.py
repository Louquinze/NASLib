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


class DARTSRandomOptimizer(DARTSOptimizer):
    """
    Implementation of the DARTS paper as in
        Liu et al. 2019: DARTS: Differentiable Architecture Search.
    """

    @staticmethod
    def update_ops(edge, topk=1):
        """
        Function to replace the primitive ops at the edges
        with the DARTS specific MixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", DARTSRandomMixedOp(primitives))

    def __init__(
            self,
            config,
            op_optimizer=torch.optim.SGD,
            arch_optimizer=torch.optim.Adam,
            loss_criteria=torch.nn.CrossEntropyLoss()
    ):
        """
        Initialize a new instance.

        Args:

        """
        super(DARTSRandomOptimizer, self).__init__(config, op_optimizer, arch_optimizer, loss_criteria)



class DARTSRandomMixedOp(DARTSMixedOp):
    """
    Continous relaxation of the discrete search space.
    """

    def __init__(self, primitives):
        super().__init__(primitives)

    def get_weights(self, edge_data):
        return edge_data.alpha, edge_data.k

    def process_weights(self, weights):
        return torch.softmax(weights, dim=-1)

    def apply_weights(self, x, weights):
        if torch.randn(1) < 0.1:
            return sum(w * op(x, None) for w, op in zip(weights, self.primitives))
        else:
            return self.primitives[torch.argmax(weights)](x, None)

