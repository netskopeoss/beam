import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .data_sources import DataSource, Mapping

"""Constants that define the limits of what can be requested from an LLM in
each request. These are just defaults in case nothing specific is defined
for the specific LLM.
"""
DEFAULT_LLM_USER_AGENT_LIMIT = 15
DEFAULT_LLM_INPUT_PREFIX = (
    """Execute all of the instructions above for the following user agent strings:"""
)
LLM_RESPONSE_FORMAT = """
Each mapping is a JSON object like this:
{
    "user_agent_string": str,
    "version": str,
    "application": {
        "name": str,
        "vendor": str,
        "description": str
    },
    "operatingsystem": {
        "name": str
    }
}
The response is a JSON object with a list of mappings:
{
    "mapping_results": [
        mappings
    ]
}
"""
DEFAULT_LLM_PROMPT = """
    Parse the given user agent strings to determine what application and
    operating system it represents. Produce a JSON formatted response
    that includes the original user agent string, version, vendor, application
    name, description of the application, and the operating system name.
    Use a human readable version of the application name if you can resolve it.
    If the string doesn't indicate a readable name, but you can infer it,
    then use that. Save it to the application field. Do not use the exact user agent
    as the application name. If you are going to use the user agent as the name of the
    application, then print "unknown" instead. Provide the vendor of the software.
    If the vendor is not indicated, try to figure out what vendor it is.
    If you can not determine the vendor, then populate the vendor field
    with a value of unknown.
    If a version number for the software is not available, then return a value of unknown.
    If there is more than one software indicated, then use the most specific
    name. For example, if it's Microsoft Office and Outlook, then tell
    me it's Outlook. Another example is if it says Cortana, then it's Cortana.
    If you have a likely match, then just use it. If there is a code name
    for the software or version, then try to decode that in the output.
    If the "application" name is abbreviated, then expand the abbreviation
    and use that instead. For example, if the name is "cpprestsdk",
    but you know that means "C++ REST SDK", then use "C++ REST SDK"
    as the name. Further, if you are able to find a description of that
    software, include the description in a "description" field.
    If you can't tell what is the software, then just give the output
    of unknown in the application field.
    Provide the operating system name and version in an "os" field.
    Do not include "likely" in the response.
    Create a list of JSON results all of the user agent mappigs.
    Your output must use this schema:
    """


def create_full_prompt(prompt_string: str, input_list: List[str]) -> str:
    """Function that will combine the initial prompt and
    input list to create the full prompt for an LLM.
    """
    input_string = "\n".join(input_list)
    full_prompt = (
        prompt_string
        + "\n"
        + LLM_RESPONSE_FORMAT
        + "\n"
        + DEFAULT_LLM_INPUT_PREFIX
        + "\n"
        + input_string
    )
    return full_prompt


class LLMAuthorization(BaseModel, ABC):
    """Abstract class for handling LLM authorization.

    This ingests the LLM credentials to be used by the LLM Configuration.
    """

    api_key: str = Field(default=None)


class LLMConfiguration(BaseModel, ABC):
    """Abstract class for handling LLM configuration.

    This selects the model, adjusts any model settings,
    and pairs it all with the LLM authorization object.
    """

    llm_model_name: str = Field()
    generation_config: Dict[str, Any] = Field()
    authorization: LLMAuthorization = Field()


class LLMWorker(BaseModel, ABC):
    """Abstract class to hold each batch of LLM input and output.

    This takes the LLM configuration and sends a prompt to the LLM
    and captures the response.
    """

    index: int
    prompt_string: str
    query_input: List[str]
    configuration: Optional[LLMConfiguration]
    logger: logging.Logger
    llm_model_object: Any = Field(default=None)
    response: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def full_prompt(self) -> str:
        """
        Build a complete prompt string for the LLM by combining the base prompt
        with additional user agent input.

        Returns:
            str: The prompt that will be sent to the LLM.
        """
        return create_full_prompt(self.prompt_string, self.query_input)

    @abstractmethod
    async def run_async_prompt(self) -> None:
        """
        Send asynchronous prompts to the LLM and store the response.

        Raises:
            Exception: If an error occurs during the asynchronous prompt execution.
        """
        pass

    @abstractmethod
    def run_single_prompt(self) -> None:
        """
        Send a single prompt to the LLM and store the response.

        Raises:
            Exception: If an error occurs during the single prompt execution.
        """
        pass


class LLMWorkProcessor(BaseModel, ABC):
    """Class to examine LLM input and batch it if necessary.

    This will moderate the amount of input tokens.
    """

    delay_between_requests: int = 0.5
    prompt_string: str = Field(default=DEFAULT_LLM_PROMPT)
    query_input: List[str]
    user_agent_limit: int = Field(default=DEFAULT_LLM_USER_AGENT_LIMIT)
    configuration: LLMConfiguration
    input_queue: deque = deque()
    workers: List[LLMWorker] = []
    logger: logging.Logger
    results: List[Mapping] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def user_agents_above_limit(self) -> bool:
        """
        Check if the total query_input length exceeds the user_agent_limit.

        Returns:
            bool: True if the total number of query items is above the limit;
            otherwise False.
        """
        return len(self.query_input) > self.user_agent_limit

    @computed_field
    @property
    def full_prompt(self) -> str:
        """
        Build a complete prompt string using the default LLM prompt and
        additional user agent input.

        Returns:
            str: The prompt that will be sent to the LLM.
        """
        return create_full_prompt(self.prompt_string, self.query_input)

    @abstractmethod
    def __add_worker__(self, index: int, input: List[str]) -> None:
        """
        Abstract method to add a new worker for a given subset of user agents.

        Args:
            index (int): The worker index or identifier.
            input (List[str]): A list of user agents or strings to process.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        pass

    @abstractmethod
    def get_results(self) -> None:
        """
        Retrieve the aggregated results from the LLM queries.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        pass

    @staticmethod
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

    def __create_input_queue__(self) -> None:
        """
        Create a queue of user agent lists respecting the user_agent_limit.

        If the total number of user agents exceeds the limit, the user agents
        are chunked into smaller lists. Otherwise, a single list is used.
        """
        if self.user_agents_above_limit:
            self.input_queue = self.make_queue_of_lists(
                self.query_input, self.user_agent_limit
            )
        else:
            self.input_queue = deque()
            self.input_queue.append(self.query_input)

    def create_workers(self) -> None:
        """
        Create a queue of worker objects to query the LLM.

        Breaks the query_input into smaller chunks if necessary, and
        creates a worker for each chunk.
        """
        self.__create_input_queue__()
        index = 0
        while len(self.input_queue) > 0:
            next_list = self.input_queue.popleft()
            index += 1
            self.__add_worker__(index, input=next_list)

    def run_workers_serially(self) -> None:
        """
        Run the workers one at a time synchronously, calling run_single_prompt
        on each worker instance in sequence.
        """
        for worker in self.workers:
            worker.run_single_prompt()

    async def run_workers_async(self) -> None:
        """
        Run the workers asynchronously, scheduling each worker's run_async_prompt
        and waiting for all tasks to finish.

        Raises:
            Exception: If an error occurs during execution of any worker task.
        """
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for worker in self.workers:
                await asyncio.sleep(self.delay_between_requests)
                task = tg.create_task(worker.run_async_prompt())
                tasks.append(task)


class LLMDataSource(DataSource):
    """Class to map user agents to applications with an LLM."""

    llm_selection: str
    work_processor: LLMWorkProcessor
    parallel_processing: bool = True
    logger: logging.Logger

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run_work_processor(self) -> None:
        """
        Launch workers to query the LLM, either asynchronously (in parallel)
        or synchronously (serially), based on the parallel_processing flag.

        Raises:
            RuntimeError: If no work processor is present.
        """
        if self.work_processor:
            self.work_processor.create_workers()
            query_start = time.perf_counter()
            if self.parallel_processing:
                asyncio.run(self.work_processor.run_workers_async())
            else:
                self.work_processor.run_workers_serially()
            query_stop = time.perf_counter()
            self.query_time = query_stop - query_start
        else:
            raise RuntimeError("Error: no work processor present.")

    def __retrieve_valid_hits__(self, initial_hits: List[Mapping]) -> List[Mapping]:
        """
        Filter out invalid hits from the provided mapping list.

        Args:
            initial_hits (List[Mapping]): The list of potential mapping objects.

        Returns:
            List[Mapping]: A filtered list of valid mapping objects. Any mapping
            object deemed invalid (e.g., unknown application) is removed and
            added to misses.
        """
        for mapping in initial_hits:
            if mapping is None:
                initial_hits.remove(mapping)
            elif mapping and mapping.application.name in ("Unknown", "unknown", ""):
                self.misses.append(mapping.user_agent_string)
                initial_hits.remove(mapping)
        return initial_hits

    def __find_missing_user_agents__(self) -> List[str]:
        """
        Identify user agents that were not successfully mapped.

        Returns:
            List[str]: A list of user agent strings missing from the hits.

        Raises:
            None
        """
        query_list = self.query_input
        ua_hits = [
            mapping.user_agent_string for mapping in self.hits if mapping is not None
        ]
        for ua in query_list:
            if ua in ua_hits:
                query_list.remove(ua)
        return query_list

    def get_results(self) -> None:
        """
        Retrieve and filter the hits and categorize any unmapped user agents
        as misses.

        This populates the instance's hits and misses with data from the
        work_processor.
        """
        self.work_processor.get_results()
        initial_hits = self.work_processor.results
        if len(initial_hits) > 0:
            self.hits = self.__retrieve_valid_hits__(initial_hits)
            missed_user_agents = self.__find_missing_user_agents__()
            self.misses.extend(missed_user_agents)
