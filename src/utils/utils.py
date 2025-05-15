import sys
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
from datetime import datetime
import time
import os
import yaml
import keyring
import re

def log(message: str, level: int) -> None:
    """
    Simple logger wrapper. `level` controls verbosity.
    Used across RAG and ingestion for debug prints.
    """
    set_level = get_log_config()["verbosity_level"]
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if level <= set_level:
        print(f"{ts} [LOG verbosity:{level}] {message}")

def log_execution_time(time_start: int, description: str, level:int = 1) -> None:
    """
    Function to log the execution time of a function.
    """
    time_end = time.time()
    log(f"{description} took {time_end - time_start:.2f} seconds", level=level)


def convert_to_abs_path(rel_path: str) -> str:
    """
    Helper to turn relative paths into absolute. Eases debugging when scanning files.
    """
    return os.path.abspath(rel_path)


def get_connection_config():
    # Load config
    with open(os.path.join(os.path.dirname(__file__), '..\\..', 'config\\connection_config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_log_config():
    # Load config
    with open(os.path.join(os.path.dirname(__file__), '..\\..', 'config\\log_config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_file_age_days(file_path: str) -> int:
    """
    Get the age of a file in days
    """
    file_mod_time = os.path.getmtime(file_path)
    file_mod_time = datetime.fromtimestamp(file_mod_time)
    file_age = datetime.now() - file_mod_time
    return file_age.days


def check_cache_exists(file_path) -> bool:
    """
    Check if the VGA guide exists in the database.
    Args:
        file_path (str): Path to the PDF file.
    Returns:
        bool: True if the guide exists, False otherwise.
    """
    # Creating cache just incase it doesn't exist
    create_cache_dir()
    # check if the file is older than 7 days
    if os.path.exists(file_path):
        age = get_file_age_days(file_path)
        if age <= 7:
            return True
    return False


def create_cache_dir() -> str:
    """
    Create a cache directory if it doesn't exist.
    Returns:
        str: Path to the cache directory.
    """
    cache_dir = "data\\Vestas_RTP\\Cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


class SuppressStderr:
    """
    A class to suppress stderr output found when parsing PDFs with pdfplumber.
    Something in the lines of 'CropBox missing from /Page.'
    """
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr


def match(pattern: re.Pattern, text: str, group_index:int=0, capitalize:bool=False):
    """Uses a regex pattern in the given text and returns the matched group."""
    match = pattern.search(text)
    if match:
        value = match.group(group_index)
        return value.capitalize() if capitalize else value
    return None

