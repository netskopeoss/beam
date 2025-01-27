"""
Utility functions
"""
import json
import os
import os.path
from collections import deque
from pathlib import Path
from typing import List

def get_project_root() -> Path:
    """
    Returns the project root path.

    Args:
        None

    Returns:
        Path: The root path of the project.

    Raises:
        None
    """
    cmd = Path(__file__)
    return Path([i for i in cmd.parents if i.as_uri().endswith("src")][0]).parent

def load_json_file(file_path: str) -> dict:
    """
    Load JSON contents from a file.

    Args:
        file_path (str): The path to the JSON file to load.

    Returns:
        dict: The contents of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    with open(file_path) as _file:
        data = json.load(_file)
    return data

def safe_create_path(path: str) -> None:
    """
    Create any directories as needed.

    Args:
        path (str): The path to create.

    Returns:
        None

    Raises:
        OSError: If the directory cannot be created.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

def safe_open(path: str):
    """
    Open "path" for writing, creating any parent directories as needed.

    Args:
        path (str): The path to open.

    Returns:
        file object: The opened file object.

    Raises:
        OSError: If the file cannot be opened.
    """
    safe_create_path(path)
    return open(path, 'w')

def save_json_data(data: dict, file_path: str) -> None:
    """
    Save JSON contents to a file.

    Args:
        data (dict): The data to save.
        file_path (str): The path to the file to save.

    Returns:
        None

    Raises:
        OSError: If the file cannot be written.
    """
    with safe_open(file_path) as _file:
        json.dump(data, _file, indent=2)

def make_queue_of_lists(input_list: List[str], limit: int) -> deque:
    """
    Break a list beyond the limit into a queue of smaller lists.

    Args:
        input_list (List[str]): The list to break into smaller lists.
        limit (int): The maximum size of each smaller list.

    Returns:
        deque: A deque containing the smaller lists.

    Raises:
        None
    """
    queue = deque()
    remainder = len(input_list) % limit
    if remainder > 0:
        queue.append(input_list[:remainder])
        del input_list[:remainder]
    while len(input_list) > 0:
        queue.append(input_list[:limit])
        del input_list[:limit]
    return queue
