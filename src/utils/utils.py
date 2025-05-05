import sys
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
from datetime import datetime
from pathlib import Path
import os
import yaml
import keyring


def traverse_project(pattern: str = '*'):
    """
    Yields all Path objects under your project root matching the glob pattern.
    E.g. traverse_project('*.yaml') or traverse_project().
    """
    root = Path(__file__).resolve().parents[2]
    yield from root.rglob(pattern)


def log(message: str, level: int = 1) -> None:
    """
    Simple logger wrapper. `level` controls verbosity.
    Used across RAG and ingestion for debug prints.
    """
    if level > 0:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{ts} [LOG] {message}")


def convert_to_abs_path(rel_path: str) -> str:
    """
    Helper to turn relative paths into absolute. Eases debugging when scanning files.
    """
    return os.path.abspath(rel_path)


def get_config():
    # Load config
    with open(os.path.join(os.path.dirname(__file__), '..\\..', 'config\\connection_config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
