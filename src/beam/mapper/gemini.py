import logging
import time
import json
import google.generativeai as genai
from typing import Any, Dict, List
from pydantic import ConfigDict
from pydantic_core import ValidationError
from google.generativeai.types import (
    AsyncGenerateContentResponse,
    GenerateContentResponse,
    BlockedPromptException
    )
from google.api_core import exceptions
from json.decoder import JSONDecodeError
from beam.mapper.data_sources import (
    Application,
    Mapping,
    OperatingSystem
)
from beam.mapper.llm import (
    LLMDataSource,
    LLMWorkProcessor,
    LLMWorker
)

logger = logging.getLogger(__name__)

# Gemini specific configuration
GEMINI_USER_AGENT_LIMIT = 30
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_SETTINGS = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "response_mime_type": "application/json"
}

def convert_response_to_json(
    response: AsyncGenerateContentResponse | GenerateContentResponse
) -> List[Dict]:
    """
    Convert the Gemini LLM response to a Python list of dictionaries.

    Args:
        response (AsyncGenerateContentResponse | GenerateContentResponse):
            The response object from the Gemini LLM. Contains a JSON-formatted
            string in its 'text' attribute.

    Returns:
        List[Dict]: A list of dictionaries parsed from the 'mapping_results'
        key of the JSON response.

    Raises:
        JSONDecodeError: If the 'text' attribute cannot be decoded as JSON.
        KeyError: If 'mapping_results' is not found in the decoded JSON.
    """
    try:
        json_data = json.loads(response.text)
        if 'mapping_results' in json_data.keys():
            return json_data['mapping_results']
        else:
            raise KeyError("The JSON response from Gemini did not contain the expected key 'mapping_results'.")
    except JSONDecodeError as e:
        logger.error(f"An error occurred while decoding the JSON received from the LLM: {e}")

def convert_json_to_mappings(
    json_data: Dict
) -> Mapping:
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
        application = Application(**json_data['application'])
        if 'operatingsystem' in json_data.keys():
            os = OperatingSystem(**json_data['operatingsystem'])
        else:
            os = OperatingSystem(name="unknown")
        
        mapping = Mapping(
            user_agent_string=json_data['user_agent_string'],
            version=json_data['version'],
            application=application,
            operatingsystem=os
            )
    except ValidationError as e:
        logging.error("Unable to validate the response from the LLM, most likely a formatting problem:")
        logging.error(e.errors())
        return
    return mapping

def process_response(
    response: AsyncGenerateContentResponse | GenerateContentResponse,
    logger: logging.Logger
) -> List[Mapping]:
    """
    Process the Gemini response and convert it into a list of Mapping objects.

    Args:
        response (AsyncGenerateContentResponse | GenerateContentResponse):
            The LLM response containing JSON data in its 'text' attribute.
        logger (logging.Logger): A logger instance for logging errors.

    Returns:
        List[Mapping]: A list of successfully converted Mapping objects. If
        the response text is empty or conversion fails, returns an empty list.
    """
    if response.text:
        json_data = convert_response_to_json(response=response)
        if (json_data):
            mappings = []
            for record in json_data:
                if 'user_agent_string' and 'application' in record.keys():
                    mapping = convert_json_to_mappings(record)
                    mappings.append(mapping)
                else:
                    logger.error(
                        "The JSON response from Gemini did not contain the expected key 'user_agent_string'."
                        )
            return mappings
        else:
            logger.error("No JSON data was available to convert to mappings.")

class GeminiWorker(LLMWorker):
    """
    Asynchronous worker that sends requests to the Google Gemini API and
    processes responses. Each worker handles a subset of user agents.
    """
    api_key: str
    llm_model_name: str
    configuration: Dict[str, Any]
    response: AsyncGenerateContentResponse | GenerateContentResponse = None

    # Needed to allow the logger to be passed in
    model_config = ConfigDict(arbitrary_types_allowed=True)  
    
    def __configure_model__(self) -> genai.GenerativeModel:
        """Configure the model object.
        """
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(
            model_name=self.llm_model_name,
            generation_config=self.configuration
            )
    
    async def run_async_prompt(self) -> None:
        """
        Send the prompt asynchronously to Google Gemini and await its response.

        Raises:
            exceptions.ResourceExhausted: If the request exceeds resource limits.
            BlockedPromptException: If the Gemini API blocks the prompt.
            Exception: Catch-all for other unexpected errors.
        """
        self.logger.info(f"Worker number {self.index} launched with {len(self.query_input)} user agents.")

        model_object = self.__configure_model__()
        query_start = time.perf_counter()
        try:
            generated_response = await model_object.generate_content_async(
                self.full_prompt
                )
        except exceptions.ResourceExhausted as e:
            self.logger.error("Google Gemini Resource exhausted:", e)
            return
        except BlockedPromptException as e:
            self.logger.error("Google Gemini Prompt blocked:", e)
            return
        except Exception as e:
            prefix = "Encountered an error from Google Gemini:"
            self.logger.error(f"{prefix} {e}")
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
        self.logger.info(f"A prompt was launched with {len(self.query_input)} user agents.")
        genai.configure(api_key=self.api_key)
        model_object = genai.GenerativeModel(
            model_name=self.llm_model_name,
            generation_config=self.configuration
            )
        query_start = time.perf_counter()
        try:
            generated_response = model_object.generate_content(
                self.full_prompt
                )
        except exceptions.ResourceExhausted as e:
            self.logger.error("Google Gemini Resource exhausted:", e)
            return
        except BlockedPromptException as e:
            self.logger.error("Google Gemini Prompt blocked:", e)
            return
        except Exception as e:
            prefix = "Encountered an error from Google Gemini:"
            self.logger.error(f"{prefix} {e}")
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
    
    def __add_worker__(self, index: int, input: List[str]) -> None:
        self.workers.append(
            GeminiWorker(
                index=index,
                prompt_string=self.prompt_string,
                query_input=input,
                api_key=self.api_key,
                llm_model_name=self.llm_model_name,
                configuration=self.configuration,
                logger=self.logger
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
                self.logger.error(
                    f"Worker {worker.index} did not return a result."
                    )  

def query_gemini(
    user_agents: List[str],
    api_key: str,
    logger: logging.Logger,
    user_agent_limit: int = GEMINI_USER_AGENT_LIMIT,
    llm_model_name: str = GEMINI_MODEL,
    settings: dict = GEMINI_SETTINGS
) -> LLMDataSource:
    """
    Query the Gemini LLM for application information based on a list
    of user agent strings.

    Args:
        user_agents (List[str]): A list of user agent strings to query.
        api_key (str): The API key for authenticating with Gemini.
        logger (logging.Logger): Logger instance for logging progress and errors.
        user_agent_limit (int, optional): Max user agents per request. Defaults to GEMINI_USER_AGENT_LIMIT.
        llm_model_name (str, optional): Gemini model name. Defaults to GEMINI_MODEL.
        settings (dict, optional): Gemini settings like temperature, top_p. Defaults to GEMINI_SETTINGS.

    Returns:
        LLMDataSource: An instance containing the aggregated results from Gemini.
    """
    gemini_processor = GeminiWorkProcessor(
        query_input=user_agents,
        api_key=api_key,
        llm_model_name=llm_model_name,
        configuration=settings,
        user_agent_limit=user_agent_limit,
        logger=logger
        )
    gemini = LLMDataSource(
        llm_selection="Gemini",
        work_processor=gemini_processor,
        logger=logger
        )
    gemini.run_work_processor()
    gemini.get_results()
    return gemini
