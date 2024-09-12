import logging
from utils.config_manager import config

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, config.logging['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=config.logging['file'],
        filemode='a'
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, config.logging['level']))
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger(__name__)

logger = setup_logging()