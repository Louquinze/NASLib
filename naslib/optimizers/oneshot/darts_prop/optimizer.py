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
import numpy as np

logger = logging.getLogger(__name__)


class DARTSPropOptimizer(DARTSOptimizer):
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
        edge.data.set("op", DARTSTPropMixedOp(primitives))

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
        super(DARTSPropOptimizer, self).__init__(config, op_optimizer, arch_optimizer, loss_criteria)


class DARTSTPropMixedOp(DARTSMixedOp):
    """
    Continous relaxation of the discrete search space.
    """

    def __init__(self, primitives):
        super().__init__(primitives)

    def apply_weights(self, x, weights):
        res = None
        rand_values = torch.rand(len(self.primitives)).cuda()
        # logger.info(
        #     f"random choice: {[rand_value < w for rand_value, w in zip(weights, rand_values) if w > rand_value]}")
        norm = sum([w for rand_value, w in zip(weights, rand_values) if w > rand_value])
        for w, op, rand_value in zip(weights, self.primitives, rand_values):
            if w > rand_value:
                if res is None:
                    res = w / norm * op(x, None)
                else:
                    res += w / norm * op(x, None)

        if res is None:
            with torch.autograd.no_grad():
                res = torch.zeros_like(self.primitives[0](x, None))

        return res
