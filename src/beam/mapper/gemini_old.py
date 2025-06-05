"""Google Gemini Module"""

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
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None
from google.api_core import exceptions
from google.generativeai.types import (
    AsyncGenerateContentResponse,
    BlockedPromptException,
    GenerateContentResponse,
)
from pydantic import ConfigDict
from pydantic_core import ValidationError

from beam.mapper.data_sources import Application, Mapping, OperatingSystem
from .llm import LLMDataSource, LLMWorker, LLMWorkProcessor

logger = logging.getLogger(__name__)

# Gemini specific configuration
GEMINI_USER_AGENT_LIMIT = 30
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_SETTINGS = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "response_mime_type": "application/json",
}


def convert_response_to_json(
    response: AsyncGenerateContentResponse | GenerateContentResponse,
) -> Optional[List[Dict]]:
    """
    Convert the Gemini LLM response to a Python list of dictionaries.

    Args:
        response (AsyncGenerateContentResponse | GenerateContentResponse):
            The response object from the Gemini LLM. Contains a JSON-formatted
            string in its 'text' attribute.        Returns:
            Optional[List[Dict]]: A list of dictionaries parsed from the 'mapping_results'
            key of the JSON response, or None if parsing fails.

    Raises:
        JSONDecodeError: If the 'text' attribute cannot be decoded as JSON.
        KeyError: If 'mapping_results' is not found in the decoded JSON.
    """
    try:
        json_data = json.loads(response.text)
        if "mapping_results" in json_data.keys():
            return json_data["mapping_results"]
        else:
            raise KeyError(
                "The JSON response from Gemini did not contain the expected key 'mapping_results'."
            )
    except JSONDecodeError as e:
        logger.error(
            "An error occurred while decoding the JSON received from the LLM: %s", e
        )


def convert_json_to_mappings(json_data: Dict) -> Optional[Mapping]:
    """
    Convert a single JSON record into a Mapping object.

    Args:
        json_data (Dict): A dictionary describing user agent data with
            'application' and optionally 'operatingsystem' keys.

    Returns:
        Optional[Mapping]: A Mapping object populated with the user_agent_string,
        version, Application, and OperatingSystem objects, or None if validation fails.

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
        return mapping
    except ValidationError as e:
        logging.error(
            "Unable to validate the response from the LLM, most likely a formatting problem:"
        )
        logging.error(e.errors())
        return None


def process_response(
    response: AsyncGenerateContentResponse | GenerateContentResponse,
    response_logger: logging.Logger,
) -> List[Mapping]:
    """
    Process the Gemini response and convert it into a list of Mapping objects.

    Args:
        response (AsyncGenerateContentResponse | GenerateContentResponse):
            The LLM response containing JSON data in its 'text' attribute.
        response_logger (logging.Logger): A logger instance for logging errors.

    Returns:
        List[Mapping]: A list of successfully converted Mapping objects. If
        the response text is empty or conversion fails, returns an empty list.
    """
    if response.text:
        json_data = convert_response_to_json(response=response)
        if json_data:
            mappings = []
            for record in json_data:
                if "user_agent_string" and "application" in record.keys():
                    mapping = convert_json_to_mappings(record)
                    if mapping:  # Only append if mapping is not None
                        mappings.append(mapping)
                else:
                    logger.error(
                        "The JSON response from Gemini did not contain the expected key 'user_agent_string'."
                    )
            return mappings
        else:
            logger.error("No JSON data was available to convert to mappings.")
    return []


class GeminiWorker(LLMWorker):
    """
    Asynchronous worker that sends requests to the Google Gemini API and
    processes responses. Each worker handles a subset of user agents.
    """

    api_key: str
    llm_model_name: str
    gemini_config: Dict[str, Any]
    response: Optional[AsyncGenerateContentResponse | GenerateContentResponse] = None

    # Needed to allow the logger to be passed in
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __configure_model__(self) -> Any:
        """Configure the model object."""
        if genai is None:
            raise ImportError("Google Generative AI library is not available")
        
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(
            model_name=self.llm_model_name, generation_config=self.gemini_config
        )

    async def run_async_prompt(self) -> None:
        """
        Send the prompt asynchronously to Google Gemini and await its response.

        Raises:
            exceptions.ResourceExhausted: If the request exceeds resource limits.
            BlockedPromptException: If the Gemini API blocks the prompt.
            Exception: Catch-all for other unexpected errors.
        """
        self.logger.info(
            f"Worker number {self.index} launched with {len(self.query_input)} user agents."
        )

        model_object = self.__configure_model__()
        query_start = time.perf_counter()
        try:
            generated_response = await model_object.generate_content_async(
                self.full_prompt
            )
        except exceptions.ResourceExhausted as e:
            self.logger.error("Google Gemini Resource exhausted: %s", e)
            return
        except BlockedPromptException as e:
            self.logger.error("Google Gemini Prompt blocked: %s", e)
            return
        except (exceptions.InternalServerError, exceptions.ServiceUnavailable) as e:
            prefix = "Encountered an error from Google Gemini:"
            self.logger.error("%s %s", prefix, e)
            return
        # Compute the query time
        query_stop = time.perf_counter()
        query_time = query_stop - query_start
        # Save the response object
        if generated_response:
            self.response = generated_response
        self.logger.info(f"Worker {self.index} completed in {query_time} seconds.")

    def run_single_prompt(self) -> None:
        """
        Send the prompt synchronously to Google Gemini and store its response.

        Raises:
            exceptions.ResourceExhausted: If the request exceeds resource limits.
            BlockedPromptException: If the Gemini API blocks the prompt.
            Exception: Catch-all for other unexpected errors.
        """
        self.logger.info(
            f"A prompt was launched with {len(self.query_input)} user agents."
        )
        genai.configure(api_key=self.api_key)
        model_object = genai.GenerativeModel(
            model_name=self.llm_model_name, generation_config=self.gemini_config
        )
        query_start = time.perf_counter()
        try:
            generated_response = model_object.generate_content(self.full_prompt)
        except exceptions.ResourceExhausted as e:
            self.logger.error("Google Gemini Resource exhausted: %s", e)
            return
        except BlockedPromptException as e:
            self.logger.error("Google Gemini Prompt blocked: %s", e)
            return
        except (exceptions.InternalServerError, exceptions.ServiceUnavailable) as e:
            prefix = "Encountered an error from Google Gemini:"
            self.logger.error("%s %s", prefix, e)
            return
        # Compute the query time
        query_stop = time.perf_counter()
        query_time = query_stop - query_start
        # Save the response object
        if generated_response:
            self.results = generated_response
        self.logger.info(f"Worker {self.index} completed in {query_time} seconds.")


class GeminiWorkProcessor(LLMWorkProcessor):
    """
    Manages concurrent requests to Google Gemini by creating multiple
    GeminiWorker instances and aggregating their responses.
    """

    api_key: str
    llm_model_name: str
    configuration: Dict[str, Any]
    user_agent_limit: int = GEMINI_USER_AGENT_LIMIT

    def __add_worker__(self, index: int, input_data: List[str]) -> None:
        self.workers.append(
            GeminiWorker(
                index=index,
                prompt_string=self.prompt_string,
                query_input=input_data,
                api_key=self.api_key,
                llm_model_name=self.llm_model_name,
                gemini_config=self.configuration,
                configuration=None,  # Use None to satisfy base class
                logger=self.logger,
            )
        )

    def get_results(self) -> None:
        """
        Retrieve and aggregate Mapping objects from each GeminiWorker's response.

        Logs errors when a worker has no response. Otherwise, extends the main
        result set with the mappings from each worker.
        """
        self.logger.info("Aggregating the results from the LLM workers.")
        for worker in self.workers:
            if worker.response:
                mappings = process_response(worker.response, self.logger)
                self.results.extend(mappings)
            else:
                self.logger.error(f"Worker {worker.index} did not return a result.")


def query_gemini(
    user_agents: List[str],
    api_key: str,
    gemini_logger: logging.Logger,
    user_agent_limit: int = GEMINI_USER_AGENT_LIMIT,
    llm_model_name: str = GEMINI_MODEL,
    settings: Optional[Dict[str, Any]] = None,
) -> LLMDataSource:
    """
    Query the Gemini LLM for application information based on a list
    of user agent strings.

    Args:
        user_agents (List[str]): A list of user agent strings to query.
        api_key (str): The API key for authenticating with Gemini.
        gemini_logger (logging.Logger): Logger instance for logging progress and errors.
        user_agent_limit (int, optional): Max user agents per request. Defaults to GEMINI_USER_AGENT_LIMIT.
        llm_model_name (str, optional): Gemini model name. Defaults to GEMINI_MODEL.
        settings (Optional[Dict[str, Any]], optional): Gemini settings like temperature, top_p. Defaults to None.

    Returns:
        LLMDataSource: An instance containing the aggregated results from Gemini.
    """
    if settings is None:
        settings = GEMINI_SETTINGS.copy()
        
    gemini_processor = GeminiWorkProcessor(
        query_input=user_agents,
        api_key=api_key,
        llm_model_name=llm_model_name,
        configuration=settings,
        user_agent_limit=user_agent_limit,
        logger=gemini_logger,
    )
    gemini = LLMDataSource(
        llm_selection="Gemini", work_processor=gemini_processor, logger=gemini_logger
    )
    gemini.run_work_processor()
    gemini.get_results()
    return gemini
