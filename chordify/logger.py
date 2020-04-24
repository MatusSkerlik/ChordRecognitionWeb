import logging


def log(f, msg, level=logging.INFO):
    # __name__ is chordify.logger
    logger = logging.getLogger(str(__name__).split('.')[0]).getChild(f.__name__)
    logger.log(level, msg)
