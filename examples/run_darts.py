import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch, DARTSTopKOptimizer, DARTSPropOptimizer
from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args

config = get_config_from_args()  # use --help so see the options
config.search.batch_size = 256
config.search.epochs = 2
config.evaluation.epochs = 2
config.save_arch_weights = True
config.save_arch_weights_path = f"{config.save}/save_arch"
set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose

search_space = SimpleCellSearchSpace() # DartsSearchSpace()  # use SimpleCellSearchSpace() for less heavy search
# search_space = SimpleCellSearchSpace()  # use SimpleCellSearchSpace() for less heavy search
search_space.top_k = 2  # Todo hacky fix this in the search space def

# optimizer = DARTSTopKOptimizer(config)
# optimizer = DARTSPropOptimizer(config)
optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search()  # Search for an architecture
trainer.evaluate(retrain=True)  # Evaluate the best architecture
