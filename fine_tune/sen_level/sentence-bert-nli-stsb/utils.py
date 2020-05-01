'''utils for saving and loading.'''
import glob
import pickle
import os
import re
import sys
import shutil
import logging

from datetime import datetime

def init_logger(name):
    '''set up training logger.'''
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    h = logging.StreamHandler(sys.stdout)
    h.flush = sys.stdout.flush
    logger.addHandler(h)
    return logger

# setup logger
logger = init_logger(__name__)
