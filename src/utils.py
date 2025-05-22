import os
import json
import logging
import pandas as pd
import numpy as np
import torch
import random
from typing import Tuple, List, Dict, Any

def load_sentiment_data(dataset_name: str, save_dir: str = "datasets") -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load sentiment data from CSV files.

    Args:
        dataset_name (str): The name of the dataset.
        save_dir (str): The directory where the dataset files are stored.

    Returns:
        Tuple[List[str], List[int], List[str], List[int]]: texts_train, labels_train, texts_test, labels_test
    """
    if dataset_name in ['arabic', 'english', 'french', 'german', 'hindi', 'italian', 'portuguese', 'spanish']:
        train_path = os.path.join(save_dir+'/multilingual/'+dataset_name+'/', f"tweet_sentiment_multilingual_train.csv")
        test_path = os.path.join(save_dir+'/multilingual/'+dataset_name+'/', f"tweet_sentiment_multilingual_test.csv")
    else:
        # Define file paths for train and test splits
        train_path = os.path.join(save_dir, f"{dataset_name}_train.csv")
        test_path = os.path.join(save_dir, f"{dataset_name}_test.csv")

    # Check if the files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        error_msg = f"Train or test files for {dataset_name} not found in {save_dir}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Load the datasets from CSV files
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    except Exception as e:
        logging.error(f"Error reading CSV files for dataset {dataset_name}: {e}")
        raise

    # Validate that required columns are present
    required_columns = {'text', 'label'}
    if not required_columns.issubset(train_data.columns) or not required_columns.issubset(test_data.columns):
        error_msg = f"Dataset {dataset_name} must contain 'text' and 'label' columns."
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Extract text and label columns
    try:
        texts_train = train_data['text'].astype(str).tolist()
        labels_train = train_data['label'].tolist()
        texts_test = test_data['text'].astype(str).tolist()
        labels_test = test_data['label'].tolist()
    except Exception as e:
        logging.error(f"Error processing data columns for dataset {dataset_name}: {e}")
        raise

    return texts_train, labels_train, texts_test, labels_test

# Custom JSON encoder for NumPy data types
class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder to handle NumPy data types.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)

def save_checkpoint(checkpoint_path: str, checkpoint: Dict[str, Any]) -> None:
    """
    Save the checkpoint to a JSON file.

    Args:
        checkpoint_path (str): The directory path to save the checkpoint.
        checkpoint (dict): The checkpoint data to save.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_path, "checkpoint.json")
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, cls=NumpyEncoder, indent=4)
        logging.info(f"Saved checkpoint at {checkpoint_file}.")
    except Exception as e:
        logging.error(f"Error saving checkpoint to {checkpoint_file}: {e}")

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load the checkpoint from a JSON file.

    Args:
        checkpoint_path (str): The directory path where the checkpoint is saved.

    Returns:
        dict: The checkpoint data.
    """
    checkpoint_file = os.path.join(checkpoint_path, "checkpoint.json")
    if not os.path.exists(checkpoint_file):
        logging.info(f"No checkpoint found at {checkpoint_file}. Starting from scratch.")
        return {}
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        logging.info(f"Loaded checkpoint from {checkpoint_file}.")
        return checkpoint
    except Exception as e:
        logging.error(f"Error loading checkpoint from {checkpoint_file}: {e}")
        return {}

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    # Setting environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    logging.info(f"Random seed set to: {seed}")


def concatenate_csv_files(input_folder, output_file):
    # Get a list of all CSV files in the input folder
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

    # Create an empty list to hold dataframes
    dataframes = []

    # Loop through the list of CSV files and read each one into a dataframe
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all dataframes into a single dataframe
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Save the concatenated dataframe to the output CSV file
    concatenated_df.to_csv(f"{input_folder}/{output_file}", index=False)
    print(f"Concatenated CSV saved to {output_file}")
