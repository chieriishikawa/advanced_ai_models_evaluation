import gc
import json
import os
import platform
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """Fix the random seed for python, numpy and torch so results can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return "cuda" when the NVIDIA GPU (RTX 4060) is available, otherwise "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype(device):
    """Use half precision on CUDA to save GPU memory, and full precision on the CPU."""
    if device == "cuda":
        return torch.float16
    return torch.float32


def free_memory():
    """Release GPU memory after a model is deleted."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_api_key(name):
    """Read an API key from the environment (loaded from the .env file)."""
    key = os.environ.get(name)
    if key:
        return key
    raise KeyError(f"{name} not found. Add it to the .env file.")


def make_folder(path):
    """Create a folder (and its parent folders) if it does not exist yet."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df, path):
    """Save a dataframe as a CSV file."""
    path = Path(path)
    make_folder(path.parent)
    df.to_csv(path, index=False)
    return path


def save_json(data, path):
    """Save a dictionary as a JSON file."""
    path = Path(path)
    make_folder(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return path


def load_json(path):
    """Read a JSON file into a dictionary."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def log_environment(path):
    """Save the python, torch and hardware details used for a run, as evidence for reproducibility."""
    device = get_device()
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if device == "cuda" else None,
    }
    save_json(info, path)
    return info
