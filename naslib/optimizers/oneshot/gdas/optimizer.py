import logging

import torch

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer

logger = logging.getLogger(__name__)


class GDASOptimizer(DARTSOptimizer):
    """
    Implements GDAS as defined in
        Dong and Yang (2019): Searching for a Robust Neural Architecture in Four GPU Hours
    """

    def __init__(
        self,
        config,
        op_optimizer: torch.optim.Optimizer = torch.optim.SGD,
        arch_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Instantiate the optimizer
        Args:
            epochs (int): Number of epochs. Required for tau
            tau_max (float): Initial tau
            tau_min (float): The minimum tau where it is decayed to
            op_optimizer (torch.optim.Optimizer): optimizer for the op weights
            arch_optimizer (torch.optim.Optimizer): optimizer for the architecture weights
            loss_criteria: The loss.
            grad_clip (float): Clipping of the gradients. Default None.
        """
        super().__init__(config, op_optimizer, arch_optimizer, loss_criteria)

        self.arch_optimizer_func = self.arch_optimizer
        self.min_optimizer_func = self.arch_optimizer

        self.epochs = config.search.epochs
        self.tau_max = torch.Tensor([config.search.tau_max])
        self.tau_min = config.search.tau_min

        # Linear tau schedule
        self.tau_step = (self.tau_min - self.tau_max) / self.epochs
        self.tau_curr = torch.Tensor([self.tau_max])  # make it checkpointable

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the GDAS specific GDASMixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", GDASMixedOp(primitives))

    def adapt_search_space(self, search_space, scope=None):
        """
        Same as in darts with a different mixop.
        Just add tau as buffer so it is checkpointed.
        """
        super().adapt_search_space(search_space, scope)
        self.graph.register_buffer("tau", self.tau_curr)

    def new_epoch(self, epoch):
        """
        Update the tau softmax parameter at the edges.
        This is also initially called before epoch 1.
        """
        super().new_epoch(epoch)

        self.tau_curr += self.tau_step
        logger.info("tau {}".format(self.tau_max))

    @staticmethod
    def reset_alphas(edge, weights):
        # sampled_arch_weight = torch.nn.functional.gumbel_softmax(
        #     edge.data.alpha, tau=float(tau), hard=True
        # )
        # edge.data.set('sampled_arch_weight', sampled_arch_weight, shared=True)

        # from gdas repo
        # https://github.com/D-X-Y/AutoDL-Projects/blob/befa6bcb00e0a8fcfba447d2a1348202759f58c9/lib/models/cell_searchs/search_model_gdas.py#L88
        # https://github.com/D-X-Y/AutoDL-Projects/blob/befa6bcb00e0a8fcfba447d2a1348202759f58c9/lib/models/cell_searchs/search_cells.py#L51
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if edge.head == 1 and edge.tail == 2:
            weight_idx = 0
        elif edge.head == 1 and edge.tail == 3:
            weight_idx = 1
        elif edge.head == 1 and edge.tail == 8:
            weight_idx = 2
        elif edge.head == 4 and edge.tail == 5:
            weight_idx = 3
        elif edge.head == 6 and edge.tail == 7:
            weight_idx = 4
        elif edge.head == 9 and edge.tail == 10:
            weight_idx = 5
        else:
            raise ValueError

        if edge.data.has("alpha"):
            edge.data.remove("alpha")
        edge.data.set("alpha", torch.nn.Parameter(weights[weight_idx]), shared=True)


    @staticmethod
    def sample_alphas(edge, tau):
        # sampled_arch_weight = torch.nn.functional.gumbel_softmax(
        #     edge.data.alpha, tau=float(tau), hard=True
        # )
        # edge.data.set('sampled_arch_weight', sampled_arch_weight, shared=True)

        # from gdas repo
        # https://github.com/D-X-Y/AutoDL-Projects/blob/befa6bcb00e0a8fcfba447d2a1348202759f58c9/lib/models/cell_searchs/search_model_gdas.py#L88
        # https://github.com/D-X-Y/AutoDL-Projects/blob/befa6bcb00e0a8fcfba447d2a1348202759f58c9/lib/models/cell_searchs/search_cells.py#L51
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arch_parameters = torch.unsqueeze(edge.data.alpha, dim=0)
        if tau == 10:
            # update scheduler etc
            if edge.data.has("alpha"):
                edge.data.remove("alpha")
            edge.data.set("alpha", torch.nn.Parameter(torch.zeros_like(arch_parameters[0])), shared=True)

        # todo reset alphas
        if tau is False:
            arch_parameters = torch.softmax(arch_parameters[0], -1)
            edge.data.set("sampled_arch_weight", torch.tensor([1. if i == torch.max(arch_parameters) else 0.
                                                               for i in arch_parameters]), shared=True)
            edge.data.set("argmax", torch.argmax(arch_parameters[0]), shared=True)
        else:
            while True:
                gumbels = -torch.empty_like(arch_parameters).exponential_().log()
                gumbels = gumbels.to(device)
                tau = tau.to(device)
                arch_parameters = arch_parameters.to(device)
                logits = (arch_parameters.log_softmax(dim=1) + gumbels) / tau
                probs = torch.nn.functional.softmax(logits, dim=1)
                index = probs.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if (
                    (torch.isinf(gumbels).any())
                    or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
                ):
                    continue
                else:
                    break

            weights = hardwts[0]
            argmaxs = index[0].item()

            edge.data.set("sampled_arch_weight", weights, shared=True)
            edge.data.set("argmax", argmaxs, shared=True)

    @staticmethod
    def remove_sampled_alphas(edge):
        if edge.data.has("sampled_arch_weight"):
            edge.data.remove("sampled_arch_weight")

    def step(self, data_train, data_val, best_model_loss, epoch):
        input_train, target_train = data_train
        input_val, target_val = data_val

        arch_weights = [i.detach() for i in self.architectural_weights.parameters()]

        # sample alphas and set to edges
        c = 1
        while True:
            # if c < 100:
            if c == 100:
                logger.info(f"reset arch weights")

                self.graph.update_edges(
                    update_func=lambda edge: self.reset_alphas(edge, arch_weights),
                    scope=self.scope,
                    private_edge_data=False,
                )

                self.architectural_weights = torch.nn.ParameterList()
                for alpha in self.graph.get_all_edge_data("alpha"):
                    self.architectural_weights.append(alpha)

                self.arch_optimizer = self.arch_optimizer_func(
                    self.architectural_weights.parameters(),
                    lr=self.config.search.arch_learning_rate,
                    betas=(0.5, 0.999),
                    weight_decay=self.config.search.arch_weight_decay * 0,
                )

                self.min_optimizer = self.min_optimizer_func(
                    self.architectural_weights.parameters(),
                    lr=self.config.search.arch_learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=self.config.search.arch_weight_decay * 0,
                )
                logger.info(arch_weights)
                logger.info(list(self.architectural_weights.parameters()))
                break

            self.graph.update_edges(
                update_func=lambda edge: self.sample_alphas(edge, torch.tensor([float(c)])),
                scope=self.scope,
                private_edge_data=False,
            )

            # Update architecture weights
            self.arch_optimizer.zero_grad()
            logits_val = self.graph(input_val)
            val_loss = self.loss(logits_val, target_val)
            if c % 100 == 99:
                logger.info(f"val_loss, tau_curr: {val_loss}, {torch.tensor([float(c)])}")
            val_loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.architectural_weights.parameters(), 5
                )
            self.arch_optimizer.step()

            self.graph.update_edges(
                update_func=lambda edge: self.sample_alphas(edge, False),
                scope=self.scope,
                private_edge_data=False,
            )

            # todo add no grad
            logits_val = self.graph(input_val)
            val_loss = self.loss(logits_val, target_val)

            if val_loss < 2.4 and val_loss < best_model_loss:
                break
            c += 1

        # Update op weights
        # while True:
        if val_loss < best_model_loss:
            self.graph.update_edges(
                update_func=lambda edge: self.sample_alphas(edge, False),
                scope=self.scope,
                private_edge_data=False,
            )

            self.op_optimizer.zero_grad()
            logits_train = self.graph(input_train)
            train_loss = self.loss(logits_train, target_train)
            train_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
            self.op_optimizer.step()

            # if train_loss < 2.4 and val_loss < best_model_loss:
            #     break

            # in order to properly unparse remove the alphas again
            self.graph.update_edges(
                update_func=self.remove_sampled_alphas,
                scope=self.scope,
                private_edge_data=False,
            )
        else:
            logits_train, train_loss = logits_val, val_loss

        return logits_train, logits_val, train_loss, val_loss, best_model_loss

@staticmethod
def add_alphas(edge):
    super().add_alphas(edge)


class GDASMixedOp(MixedOp):
    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed op for GDAS.
        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(primitives)
        self.min_cuda_memory = min_cuda_memory

    def get_weights(self, edge_data):
        return edge_data.sampled_arch_weight

    def process_weights(self, weights):
        return weights

    def apply_weights(self, x, weights):
        """
        Applies the gumbel softmax to the architecture weights
        before forwarding `x` through the graph as in DARTS
        """

        argmax = torch.argmax(weights)

        weighted_sum = sum(
            weights[i] * op(x, None) if i == argmax else weights[i]
            for i, op in enumerate(self.primitives)
        )

        return weighted_sum
