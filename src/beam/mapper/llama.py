"""Llama Local Model Module"""

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
import logging
import time
from json.decoder import JSONDecodeError
from typing import Any, Dict, List
import requests

from pydantic import ConfigDict
from pydantic_core import ValidationError

from beam.mapper.data_sources import Application, Mapping, OperatingSystem
from .llm import LLMDataSource, LLMWorker, LLMWorkProcessor
from beam import constants

logger = logging.getLogger(__name__)

# Llama specific configuration
LLAMA_USER_AGENT_LIMIT = 20
LLAMA_MODEL = "llama3.2:1b"
LLAMA_BASE_URL = "http://localhost:11434"
LLAMA_SETTINGS = {
    "temperature": 0.1,  # Lower temperature for more consistent output
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 2048,
}

# Llama-specific prompt that enforces JSON output
LLAMA_PROMPT = """
Parse user agent strings and return ONLY valid JSON. Do not include any explanatory text, code, or formatting.

For each user agent string, create a JSON object with this exact structure:
{
    "user_agent_string": "original string",
    "version": "version or unknown",
    "application": {
        "name": "application name or unknown",
        "vendor": "vendor name or unknown", 
        "description": "description or unknown"
    },
    "operatingsystem": {
        "name": "OS name or unknown"
    }
}

Return an array of these objects. Example output:
[
    {
        "user_agent_string": "Chrome/91.0.4472.124",
        "version": "91.0.4472.124",
        "application": {
            "name": "Chrome",
            "vendor": "Google",
            "description": "Web browser"
        },
        "operatingsystem": {
            "name": "unknown"
        }
    }
]

IMPORTANT: Your response must be ONLY valid JSON, no other text.
"""


def convert_response_to_json(response_text: str) -> List[Dict]:
    """
    Convert the Llama LLM response to a Python list of dictionaries.

    Args:
        response_text (str): The response text from the Llama LLM containing JSON data.

    Returns:
        List[Dict]: A list of dictionaries parsed from the 'mapping_results'
        key of the JSON response.

    Raises:
        JSONDecodeError: If the response text cannot be decoded as JSON.
        KeyError: If 'mapping_results' is not found in the decoded JSON.
    """
    try:
        # Try to find and parse different JSON formats that Llama might return
        json_objects = []
        
        # Look for JSON arrays first (preferred format)
        array_start = response_text.find('[')
        if array_start != -1:
            array_end = response_text.rfind(']') + 1
            if array_end > array_start:
                try:
                    array_str = response_text[array_start:array_end]
                    json_objects = json.loads(array_str)
                    if isinstance(json_objects, list):
                        # Log the successful mappings
                        for obj in json_objects:
                            if "user_agent_string" in obj and "application" in obj:
                                app_name = obj["application"].get("name", "unknown")
                                user_agent = obj["user_agent_string"]
                                logger.info(f"Llama mapped '{user_agent}' to application '{app_name}'")
                        return json_objects
                except JSONDecodeError:
                    logger.warning("Failed to parse JSON array, trying object format")
        
        # Look for JSON object with mapping_results key
        obj_start = response_text.find('{')
        if obj_start != -1:
            obj_end = response_text.rfind('}') + 1
            if obj_end > obj_start:
                try:
                    obj_str = response_text[obj_start:obj_end]
                    json_data = json.loads(obj_str)
                    
                    # Check if it has mapping_results key
                    if "mapping_results" in json_data:
                        return json_data["mapping_results"]
                    
                    # If it's a single mapping object, wrap it in a list
                    if "user_agent_string" in json_data:
                        app_name = json_data.get("application", {}).get("name", "unknown")
                        user_agent = json_data["user_agent_string"]
                        logger.info(f"Llama mapped '{user_agent}' to application '{app_name}'")
                        return [json_data]
                        
                except JSONDecodeError:
                    logger.warning("Failed to parse JSON object")
        
        # If we get here, no valid JSON was found
        raise JSONDecodeError("No valid JSON found in response", response_text, 0)
    except JSONDecodeError as e:
        logger.error(
            f"An error occurred while decoding the JSON received from the LLM: {e}"
        )
        raise


def convert_json_to_mappings(json_data: Dict) -> Mapping:
    """
    Convert a single JSON record into a Mapping object.

    Args:
        json_data (Dict): A dictionary describing user agent data with
            'application' and optionally 'operatingsystem' keys.

    Returns:
        Mapping: A Mapping object populated with the user_agent_string,
        version, Application, and OperatingSystem objects.

    Raises:
        ValidationError: If the passed data cannot be validated as a valid
        Mapping, Application, or OperatingSystem.
    """
    try:
        # Convert the JSON to a Mapping.
        application = Application(**json_data["application"])
        if "operatingsystem" in json_data.keys():
            os = OperatingSystem(**json_data["operatingsystem"])
        else:
            os = OperatingSystem(name="unknown")

        mapping = Mapping(
            user_agent_string=json_data["user_agent_string"],
            version=json_data["version"],
            application=application,
            operatingsystem=os,
        )
    except ValidationError as e:
        logging.error(
            "Unable to validate the response from the LLM, most likely a formatting problem:"
        )
        logging.error(e.errors())
        return None
    return mapping


def process_response(response_text: str, logger: logging.Logger) -> List[Mapping]:
    """
    Process the Llama response and convert it into a list of Mapping objects.

    Args:
        response_text (str): The LLM response containing JSON data.
        logger (logging.Logger): A logger instance for logging errors.

    Returns:
        List[Mapping]: A list of successfully converted Mapping objects. If
        the response text is empty or conversion fails, returns an empty list.
    """
    if response_text:
        try:
            json_data = convert_response_to_json(response_text)
            if json_data:
                mappings = []
                for record in json_data:
                    if "user_agent_string" in record.keys() and "application" in record.keys():
                        mapping = convert_json_to_mappings(record)
                        if mapping:
                            mappings.append(mapping)
                    else:
                        logger.error(
                            "The JSON response from Llama did not contain the expected key 'user_agent_string'."
                        )
                return mappings
            else:
                logger.error("No JSON data was available to convert to mappings.")
                return []
        except Exception as e:
            logger.error(f"Error processing Llama response: {e}")
            return []
    return []


class LlamaWorker(LLMWorker):
    """
    Worker that sends requests to the local Llama API and processes responses.
    Each worker handles a subset of user agents.
    """

    base_url: str
    llm_model_name: str

    # Needed to allow the logger to be passed in
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __send_request__(self, prompt: str) -> str:
        """Send request to Ollama API."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.llm_model_name,
            "prompt": prompt,
            "stream": False,
            "options": LLAMA_SETTINGS  # Use the global Llama settings
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request to Llama API failed: {e}")
            return ""

    async def run_async_prompt(self) -> None:
        """
        Send the prompt asynchronously to local Llama and await its response.

        Raises:
            Exception: Catch-all for unexpected errors.
        """
        self.logger.info(
            f"Worker number {self.index} launched with {len(self.query_input)} user agents."
        )

        query_start = time.perf_counter()
        try:
            # For now, use synchronous request in async context
            # This could be improved with aiohttp for true async
            response_text = self.__send_request__(self.full_prompt)
            if response_text:
                self.response = response_text
        except Exception as e:
            self.logger.error(f"Encountered an error from Llama: {e}")
            return
        
        query_stop = time.perf_counter()
        query_time = query_stop - query_start
        self.logger.info(f"Worker {self.index} completed in {query_time} seconds.")

    def run_single_prompt(self) -> None:
        """
        Send the prompt synchronously to local Llama and store its response.

        Raises:
            Exception: Catch-all for unexpected errors.
        """
        self.logger.info(
            f"A prompt was launched with {len(self.query_input)} user agents."
        )
        
        query_start = time.perf_counter()
        try:
            response_text = self.__send_request__(self.full_prompt)
            if response_text:
                self.response = response_text
        except Exception as e:
            self.logger.error(f"Encountered an error from Llama: {e}")
            return
        
        query_stop = time.perf_counter()
        query_time = query_stop - query_start
        self.logger.info(f"Worker {self.index} completed in {query_time} seconds.")


class LlamaWorkProcessor(LLMWorkProcessor):
    """
    Manages concurrent requests to local Llama by creating multiple
    LlamaWorker instances and aggregating their responses.
    """

    base_url: str
    llm_model_name: str
    configuration: Dict[str, Any]
    user_agent_limit: int = LLAMA_USER_AGENT_LIMIT

    def __add_worker__(self, index: int, input: List[str]) -> None:
        self.workers.append(
            LlamaWorker(
                index=index,
                prompt_string=self.prompt_string,
                query_input=input,
                configuration=None,  # Required by base class LLMWorker
                response=None,  # Initialize the required response field
                base_url=self.base_url,
                llm_model_name=self.llm_model_name,
                logger=self.logger,
            )
        )

    def get_results(self) -> None:
        """
        Retrieve and aggregate Mapping objects from each LlamaWorker's response.

        Logs errors when a worker has no response. Otherwise, extends the main
        result set with the mappings from each worker.
        """
        self.logger.info("Aggregating the results from the LLM workers.")
        for worker in self.workers:
            if hasattr(worker, 'response') and worker.response:
                mappings = process_response(worker.response, self.logger)
                self.results.extend(mappings)
            else:
                self.logger.error(f"Worker {worker.index} did not return a result.")


def query_llama(
    user_agents: List[str],
    logger: logging.Logger,
    user_agent_limit: int = LLAMA_USER_AGENT_LIMIT,
    llm_model_name: str = LLAMA_MODEL,
    base_url: str = None,  # Will use environment variable if None
    settings: dict = LLAMA_SETTINGS,
) -> LLMDataSource:
    """
    Query the local Llama LLM for application information based on a list
    of user agent strings.

    Args:
        user_agents (List[str]): A list of user agent strings to query.
        logger (logging.Logger): Logger instance for logging progress and errors.
        user_agent_limit (int, optional): Max user agents per request. Defaults to LLAMA_USER_AGENT_LIMIT.
        llm_model_name (str, optional): Llama model name. Defaults to LLAMA_MODEL.
        base_url (str, optional): Base URL for Ollama API. Defaults to LLAMA_BASE_URL.
        settings (dict, optional): Llama settings like temperature, top_p. Defaults to LLAMA_SETTINGS.

    Returns:
        LLMDataSource: An instance containing the aggregated results from Llama.
    """
    # Use environment variable if base_url is not specified
    if base_url is None:
        base_url = constants.LLAMA_BASE_URL
        
    llama_processor = LlamaWorkProcessor(
        query_input=user_agents,
        prompt_string=LLAMA_PROMPT,  # Use our custom Llama prompt
        base_url=base_url,
        llm_model_name=llm_model_name,
        configuration=settings,
        user_agent_limit=user_agent_limit,
        logger=logger,
    )
    llama = LLMDataSource(
        llm_selection="Llama", work_processor=llama_processor, logger=logger
    )
    llama.run_work_processor()
    llama.get_results()
    return llama