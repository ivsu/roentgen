import os
from common.logger import Logger

logger = Logger(__name__)


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.INFO, layout='debug')
