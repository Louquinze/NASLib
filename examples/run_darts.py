import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch, DARTSTopKOptimizer, DARTSPropOptimizer, DARTSScheduledOptimizer
from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args

config = get_config_from_args()  # use --help so see the options
print(config)
config.search.batch_size = 128
config.save_arch_weights = True
config.save_arch_weights_path = f"{config.save}/save_arch"
set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose

# search_space = DartsSearchSpace()  # use SimpleCellSearchSpace() for less heavy search
search_space = SimpleCellSearchSpace()  # use SimpleCellSearchSpace() for less heavy search
# search_space.top_k = 2  # Todo hacky fix this in the search space def

config.search.epochs = 10
config.search.checkpoint_freq = config.search.epochs + 1
config.evaluation.epochs = 5
config.evaluation.checkpoint_freq = config.evaluation.epochs + 1
optimizer = GDASOptimizer(config)
# optimizer = DARTSPropOptimizer(config)
# optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search()  # Search for an architecture
# trainer.evaluate()  # Evaluate the best architecture
