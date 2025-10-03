import os
import numpy as np
import logging

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_numpy_array(file_path, data):
    np.save(file_path, data)
    logging.info(f"Saved numpy array to {file_path}")

def load_numpy_array(file_path):
    if os.path.exists(file_path):
        data = np.load(file_path)
        logging.info(f"Loaded numpy array from {file_path}")
        return data
    else:
        logging.error(f"File not found: {file_path}")
        return None

def setup_logging(log_file='process.log'):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info("Logging is set up.")