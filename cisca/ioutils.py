#import torch			 
import tensorflow as tf
import numpy as np
import random
import os
import json
import pathlib
import glob
import pickle
from cisca.augmentation.composition import Compose, OneOf

DEFAULT_RANDOM_SEED = 2024

def _raise(e):
    """Raise an exception if it is a subclass of BaseException, else raise a ValueError."""
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)

def set_seed(seed_value):
    """
    Set the seed for various random number generators to ensure reproducibility.
    
    Args:
        seed_value (int): The seed value to be used.
    """
    tf.random.set_seed(seed_value)
    tf.keras.utils.set_random_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

def load_json(fpath):
    """
    Load data from a JSON file.
    
    Args:
        fpath (str): The file path to the JSON file.
    
    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(fpath, 'r') as f:
        return json.load(f)

class MyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Compose, OneOf, and pathlib.Path objects."""

    def default(self, obj):
        if isinstance(obj, Compose) or isinstance(obj, OneOf):
            return obj.name
        if isinstance(obj, pathlib.Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def save_json(data, fpath, **kwargs):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): The data to be saved.
        fpath (str): The file path to save the JSON file.
        **kwargs: Additional arguments to pass to json.dumps().
    """
    with open(fpath, 'w') as f:
        f.write(json.dumps(data, **kwargs))

def save_pickle(fpath, array):
    """
    Save a NumPy array to a pickle file.
    
    Args:
        fpath (str): The file path to save the pickle file.
        array (np.ndarray): The NumPy array to be saved.
    """
    with open(fpath, 'wb') as f:
        pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(fpath):
    """
    Load a NumPy array from a pickle file.
    
    Args:
        fpath (str): The file path of the pickle file.
    
    Returns:
        np.ndarray: The loaded NumPy array.
    """
    with open(fpath, 'rb') as f:
        return pickle.load(f)
					   
def seed_basic(seed=DEFAULT_RANDOM_SEED):
    """
    Set the seed for basic random number generators (random, numpy, and Python's hash seed).
    
    Args:
        seed (int): The seed value to be used. Default is DEFAULT_RANDOM_SEED.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
				
def seed_tf(seed=DEFAULT_RANDOM_SEED):
    """
    Set the seed for TensorFlow random number generators.
    
    Args:
        seed (int): The seed value to be used. Default is DEFAULT_RANDOM_SEED.
    """
    tf.random.set_seed(seed)

# Uncomment the following function if you are using PyTorch.
# import torch
# def seed_torch(seed=DEFAULT_RANDOM_SEED):
#     """
#     Set the seed for PyTorch random number generators.
#     
#     Args:
#         seed (int): The seed value to be used. Default is DEFAULT_RANDOM_SEED.
#     """
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
							 
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    """
    Set the seed for all random number generators (basic, TensorFlow, and optionally PyTorch).
					
    
    Args:
        seed (int): The seed value to be used. Default is DEFAULT_RANDOM_SEED.
    """
    seed_basic(seed)
    seed_tf(seed)
    # Uncomment the following line if you are using PyTorch.
    # seed_torch(seed)

def find_most_recent_file(directory, suffix):
    """
    Find the most recent file in a directory with a given suffix.
    
    Args:
        directory (str): The directory to search in.
        suffix (str): The suffix of the files to search for.
    
    Returns:
        str: The path of the most recent file, or None if no such file exists.
    """
    files = glob.glob(os.path.join(directory, f'*{suffix}'))
    if not files:
        return None
															  
    return max(files, key=os.path.getmtime)

def list_files_with_extension(folder_path, extension, remove_suffix = True):
    files = []
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            if remove_suffix:
                file = os.path.splitext(file)[0]  # Remove the extension
            files.append(file)
    return files
						   






