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


class DARTSScheduledRevOptimizerV3(DARTSOptimizer):
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
        edge.data.set("op", DARTSScheduledMixedOp(primitives))

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
        super(DARTSScheduledRevOptimizerV3, self).__init__(config, op_optimizer, arch_optimizer, loss_criteria)
        self.epochs = config.search.epochs

    @staticmethod
    def sample_alphas(edge, epoch, max_epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # arch_parameters = torch.unsqueeze(edge.data.alpha, dim=0)
        w = np.exp(-(((epoch-max_epochs)//2)**2)//max_epochs)
        k = max(int(w * len(edge.data.alpha)), 1)
        edge.data.set("k", k, shared=True)

    @staticmethod
    def remove_sampled_alphas(edge):
        if edge.data.has("k"):
            edge.data.remove("k")

    def step(self, data_train, data_val, epoch):

        input_train, target_train = data_train
        input_val, target_val = data_val

        # sample alphas and set to edges
        self.graph.update_edges(
            update_func=lambda edge: self.sample_alphas(edge, epoch, self.epochs),
            scope=self.scope,
            private_edge_data=False,
        )

        # Update architecture weights
        self.arch_optimizer.zero_grad()
        logits_val = self.graph(input_val)
        val_loss = self.loss(logits_val, target_val)
        val_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_value_(
                self.architectural_weights.parameters(), self.grad_clip
            )
        self.arch_optimizer.step()

        # has to be done again, cause val_loss.backward() frees the gradient from sampled alphas
        # TODO: this is not how it is intended because the samples are now different. Another
        # option would be to set val_loss.backward(retain_graph=True) but that requires more memory.
        self.graph.update_edges(
            update_func=lambda edge: self.sample_alphas(edge, epoch, self.epochs),
            scope=self.scope,
            private_edge_data=False,
        )

        # Update op weights
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        # in order to properly unparse remove the alphas again
        self.graph.update_edges(
            update_func=self.remove_sampled_alphas,
            scope=self.scope,
            private_edge_data=False,
        )

        return logits_train, logits_val, train_loss, val_loss


class DARTSScheduledMixedOp(DARTSMixedOp):
    """
    Continous relaxation of the discrete search space.
    """

    def __init__(self, primitives):
        super().__init__(primitives)

    def get_weights(self, edge_data):
        return edge_data.alpha, edge_data.k

    def process_weights(self, weights):
        return weights

    def apply_weights(self, x, weights):
        arch_alphas = torch.cat(tuple(
            torch.unsqueeze(weights[0][idx], dim=0) for idx in torch.topk(weights[0], weights[1]).indices))
        arch_alphas = torch.softmax(arch_alphas, dim=-1)
        return sum(
            weight * self.primitives[int(idx)](x, None) for idx, weight in
            zip(torch.topk(weights[0], weights[1]).indices, arch_alphas))
