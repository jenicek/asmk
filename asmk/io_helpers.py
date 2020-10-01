"""Helper functions related to io"""

import logging
import pickle
import urllib.request
from pathlib import Path
import yaml

# Params

def load_params(path):
    """Return loaded parameters from a yaml file"""
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


# Logging

def init_logger(log_path):
    """Return a logger instance which logs to stdout and, if log_path is not None, also to a file"""
    logger = logging.getLogger("ASMK")
    logger.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter('%(name)s %(levelname)s: %(message)s'))
    logger.addHandler(stdout_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Load and save state dicts

def load_pickle(path):
    """Load pickled data from path"""
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def save_pickle(path, data):
    """Save data to path using pickle"""
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


# Download

def download_files(names, root_path, base_url, logfunc=None):
    """Download files from given url to given directory path. If logfunc given, use it to log
        status."""
    root_path = Path(root_path)
    for name in names:
        path = root_path / name
        if path.exists():
            continue
        if logfunc:
            logfunc(f"Downloading file '{name}'")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(base_url + name, path)
