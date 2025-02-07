import logging
import logging.config

from run import MultiHotEncoder, run

from beam.constants import LOG_CONFIG

if __name__ == "__main__":
    logging.config.fileConfig(LOG_CONFIG)
    logger = logging.getLogger("main")
    _m = MultiHotEncoder()
    run(logger=logger)
