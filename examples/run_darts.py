import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch, DARTSTopKOptimizer, DARTSPropOptimizer
from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args

config = get_config_from_args()  # use --help so see the options
config.search.batch_size = 32
set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose

search_space = DartsSearchSpace()  # use SimpleCellSearchSpace() for less heavy search
# search_space = SimpleCellSearchSpace()  # use SimpleCellSearchSpace() for less heavy search
search_space.top_k = 2  # Todo hacky fix this in the search space def

# optimizer = DARTSTopKOptimizer(config)
optimizer = DARTSPropOptimizer(config)
# optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search()  # Search for an architecture
trainer.evaluate()  # Evaluate the best architecture
