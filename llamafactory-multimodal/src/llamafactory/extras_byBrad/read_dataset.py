import os
import numpy as np
import json
from tqdm import tqdm

def read_raw_dataset(raw_data_dir):
    dataset = []
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(raw_data_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                dataset.append(data)
    return dataset

def prepare_dataset(raw_data_dir):
    raw_dataset = read_raw_dataset(raw_data_dir)
    processed_data = []
    
    for data in tqdm(raw_dataset, desc="Processing dataset"):
        # Here you can add any preprocessing steps needed for your data
        processed_data.append(data)
    
    return processed_data

def load_numpy_data(numpy_data_dir):
    numpy_files = []
    for filename in os.listdir(numpy_data_dir):
        if filename.endswith('.npy'):
            file_path = os.path.join(numpy_data_dir, filename)
            data = np.load(file_path)
            numpy_files.append(data)
    return numpy_files

def save_processed_data(processed_data, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, data in enumerate(tqdm(processed_data, desc="Saving processed data")):
        np.save(os.path.join(save_dir, f'processed_data_{i}.npy'), data)