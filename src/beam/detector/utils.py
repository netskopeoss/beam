import json
import os
import os.path
from pathlib import Path


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
    return open(path, "w")


def save_json_data(data: dict | list, file_path: str) -> None:
    """
    Save JSON contents to a file.

    Args:
        data (dict | list): The data to save.
        file_path (str): The path to the file to save.

    Returns:
        None

    Raises:
        OSError: If the file cannot be written.
    """
    with safe_open(file_path) as _file:
        json.dump(data, _file, indent=2)


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
