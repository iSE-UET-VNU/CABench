import os
import json
import tarfile
from typing import Optional, Dict
import pandas as pd
import requests
from tqdm import tqdm

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

from utils.logs import logger


def load_metadata(path: str):
    """Load metadata.json from task directory"""
    if not os.path.exists(os.path.join(path, "metadata.json")):
        raise FileNotFoundError(f"Metadata file not found in {path}")
    with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def load_task_description(path: str) -> str:
    """Load task description from task_description.txt"""
    if not os.path.exists(os.path.join(path, "task_description.txt")):
        raise FileNotFoundError(f"Description file not found in {path}")
    with open(os.path.join(path, "task_description.txt"), "r", encoding="utf-8") as f:
        task_description = f.read()
    return task_description


def load_files_in_folder(path: str) -> dict:
    """Load and categorize files by type (image, video, audio, text)"""
    extensions = {
        'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'},
        'video': {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v'},
        'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}
    }
    result = {
        "image_paths": [],
        "video_paths": [],
        "audio_paths": [],
        "text_data": []
    }    
    
    for current_dir, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(current_dir, file)
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in extensions['image']:
                result['image_paths'].append(file_path)
            elif file_extension in extensions['video']:
                result['video_paths'].append(file_path)
            elif file_extension in extensions['audio']:
                result['audio_paths'].append(file_path)
            elif file == "record.json":
                with open(file_path, "r", encoding="utf-8") as f:
                    record = json.load(f)
                    result['text_data'].append(record)
    return result


def load_data(path: str) -> dict:
    """Load data from inputs directory"""
    inputs_path = os.path.join(path, "inputs")
    if not os.path.exists(inputs_path):
        raise FileNotFoundError(f"Path not found: {path}")
    inputs = {}
    for f in os.listdir(inputs_path):
        if os.path.isdir(os.path.join(inputs_path, f)):
            inputs[f] = load_files_in_folder(os.path.join(inputs_path, f))
    return inputs


def load_label(path: str) -> pd.DataFrame:
    """Load labels.csv from path"""
    label_file = os.path.join(path, "labels.csv")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found in {path}")
    return pd.read_csv(label_file)


def load_prediction(path: str, round_num: Optional[int] = None) -> pd.DataFrame:
    """Load predictions.csv from path with optional round number"""
    if round_num is not None:
        prediction_file = os.path.join(path, f"predictions_{round_num}.csv")
    else:
        prediction_file = os.path.join(path, "predictions.csv")
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f"Prediction file not found in {path}")
    return pd.read_csv(prediction_file)


def load_task_dirs(root_path: str):
    """Load all task directories that contain metadata.json"""
    if not root_path.startswith("tasks"):
        raise ValueError(f"Path must start with 'tasks'")
    task_dirs = []
    
    for current_dir, dirs, files in os.walk(root_path):
        stop_condition = "metadata.json" in files
        if stop_condition:
            task_dirs.append(current_dir)
            dirs[:] = [] # clear dirs so os.walk doesn't recurse further
    
    return task_dirs


# Download functionality
def download_file(url: str, filename: str) -> None:
    """Download a file from Google Drive using gdown."""
    if not HAS_GDOWN:
        raise ImportError("gdown package is required for downloading. Install it with: pip install gdown")
    gdown.download(url, filename, quiet=False)


def extract_tar_gz(filename: str, extract_path: str) -> None:
    """Extract a tar.gz file to the specified path."""
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=extract_path)


def process_dataset(url: str, filename: str, extract_path: str) -> None:
    """Download, extract, and clean up a dataset."""
    logger.info(f"Downloading {filename}...")
    download_file(url, filename)

    logger.info(f"Extracting {filename}...")
    extract_tar_gz(filename, extract_path)

    logger.info(f"{filename} download and extraction completed.")

    os.remove(filename)
    logger.info(f"Removed {filename}")


def get_available_datasets() -> Dict[str, Dict[str, str]]:
    """Get dictionary of available datasets to download"""
    return {
        "datasets": {
            "url": "https://drive.google.com/uc?export=download&id=1wsU7Q2iNR0TW6QzyXe7ctvdDLQP20bIR",
            "filename": "cabench_data.tar.gz",
            "extract_path": "tasks",
        },
        "results": {
            "url": "https://drive.google.com/uc?export=download&id=1zY5jo20v0IQbmxE5CF0zxtbUvIq-PybS",
            "filename": "results.tar.gz",
            "extract_path": "results",
        }
    }


def download_datasets(required_datasets, skip_existing: bool = False):
    """Main function to download and process selected datasets"""
    datasets_to_download = get_available_datasets()
    
    if skip_existing:
        logger.info("Skip downloading datasets (--skip-existing flag set)")
        return
    
    for dataset_name in required_datasets:
        if dataset_name not in datasets_to_download:
            logger.error(f"Dataset '{dataset_name}' not found. Available datasets: {list(datasets_to_download.keys())}")
            continue
            
        dataset = datasets_to_download[dataset_name]
        extract_path = dataset["extract_path"]
        
        # Check if already exists
        if os.path.exists(extract_path) and os.listdir(extract_path):
            logger.info(f"Dataset '{dataset_name}' already exists at {extract_path}")
            continue
            
        process_dataset(dataset["url"], dataset["filename"], extract_path) 