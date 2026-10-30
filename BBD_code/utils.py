import os

import yaml
import numpy as np
import random
import torch
import logging
import logging.handlers

def fix_random(
        random_seed: int = 0
) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def generate_random_except(start, end, exclude):
    while True:
        num = random.randint(start, end)
        if num != exclude :
            return num

def setup_logger(log_file_path):
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    log_file_path = os.path.join(log_file_path, 'logs.log')

    logger = logging.getLogger('bbd_logger')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=1024*1024*10, backupCount=5)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger