"""Detector Utilities Module"""

# Copyright 2025 Netskope, Inc.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Authors:
# - Colin Estep
# - Dagmawi Mulugeta

import json
import os
import os.path
from typing import Dict, List, TextIO, Union


def safe_open(path: str) -> TextIO:
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
    return open(path, "w", encoding="utf-8")


def save_json_data(data: Union[Dict, List], file_path: str) -> None:
    """
    Save JSON contents to a file.

    Args:
        data (Dict | List): The data to save.
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


def load_json_file(file_path: str) -> Union[Dict, List]:
    """
    Load JSON contents from a file.

    Args:
        file_path (str): The path to the JSON file to load.

    Returns:
        Union[Dict, List]: The contents of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """

    with open(file_path, encoding="utf-8") as _file:
        data = json.load(_file)

    return data
