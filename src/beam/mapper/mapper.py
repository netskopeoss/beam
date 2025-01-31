import logging
import re
from typing import List

from pydantic import ConfigDict

from .agent_parser import query_agent_parser
from .data_sources import APIDataSource, DataSource
from .datastore import DataStoreHandler
from .gemini import query_gemini
from .llm import LLMDataSource


class UserAgentMapper(DataSource):
    """Class for that searches multiple data sources to
        map user agents to applications.

    This class should take the user agents we want to be mapped,
    And then handle seamlessly all of the data sources for mapping.
    It needs to collect the applications found from the user agents
    and return all of the data as one object.
    """

    llm_api_key: str
    llm_selection: str
    datastore: DataStoreHandler
    logger: logging.Logger
    llm: LLMDataSource = None
    api: APIDataSource = None

    # Needed to allow the logger to be passed in
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def search(self) -> None:
        """
        Search the data sources in order (datastore, LLM, API) for the user agents.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        if len(self.query_input) > 0:
            self.search_datastore()
            self.search_llm()
            self.search_api()

    def search_datastore(self) -> None:
        """
        Search the datastore for the user agents in self.query_input.
        Adds any found agents to self.hits and unfound ones to self.misses.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        self.datastore.search(user_agents=self.query_input)
        # Update the hits and misses lists
        if self.datastore.hits_found:
            self.logger.info(
                f"Found {len(self.datastore.hits)} user agents in the datastore."
            )
            self.hits.extend(self.datastore.hits)
        else:
            self.logger.info("No user agents were found in the datastore.")
        if self.datastore.misses_found:
            self.misses.extend(self.datastore.misses)
        else:
            self.logger.info("Nothing left to search after querying the datastore.")
            return
        self.logger.info(
            f"A total of {len(self.misses)} misses exist after checking the datastore."
        )

    def search_llm(self) -> None:
        """
        Search an LLM for any user agents that were missed by the datastore.
        Updates self.hits and self.misses accordingly.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        if self.misses == 0:
            self.logger.info("Skipping the LLM check because 0 misses are left.")
            return
        self.logger.info(
            f"Checking LLM source {self.llm_selection} for a total of {len(self.misses)} user agents."
        )
        if self.llm_api_key:
            if self.llm_selection == "Gemini":
                self.llm = query_gemini(
                    user_agents=self.misses,
                    api_key=self.llm_api_key,
                    logger=self.logger,
                )
        else:
            self.logger.error("No LLM API key was provided.")

        if self.llm and self.llm.hits_found:
            self.logger.info(
                f"{self.llm_selection} was able to map {len(self.llm.hits)} user agents."
            )
            self.hits.extend(self.llm.hits)
        else:
            self.logger.info(f"{self.llm_selection} was unable to map any user agents.")

        if self.llm and self.llm.misses_found:
            # We need to reset the misses list to the new misses
            # Some of the old misses may be hits now
            self.misses = self.llm.misses
        else:
            self.logger.info("Nothing left to search after querying the LLM.")
            return
        self.logger.info(
            f"A total of {len(self.misses)} misses exist after checking {self.llm_selection}."
        )

    def search_api(self) -> None:
        """
        Search an external API for any user agents still marked as misses.
        Updates self.hits and self.misses based on the results.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        if self.misses == 0:
            self.logger.info("Skipping the API check because 0 misses are left.")
            return
        self.logger.info(
            f"Checking the API for a total of {len(self.misses)} user agents."
        )
        self.api = query_agent_parser(self.misses)
        if self.api.hits_found:
            self.hits.extend(self.api.hits)

        if self.api.misses_found:
            # We need to reset the misses list to the new misses
            # Some of the old misses may be hits now
            self.misses = self.api.misses
        else:
            self.logger.info(
                "All user agents were mapped after checking the API data source."
            )
            return
        self.logger.info(
            f"A total of {len(self.misses)} misses exist after checking the API."
        )

    def save_results(self) -> None:
        """
        Save the successfully mapped user agents (in self.hits) to the datastore.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        if self.hits_found:
            self.logger.info(
                f"A total of {len(self.hits)} user agents were found. Writing those to the db."
            )
            self.datastore.save_results(self.hits)
        if self.misses_found:
            self.logger.info(f"The mapper missed {len(self.misses)} user agents.")


def query_user_agent_mapper(
    user_agents: List[str],
    db_path: str,
    logger: logging.Logger,
    llm_api_key: str = None,
    llm_selection: str = "Gemini",
) -> UserAgentMapper:
    """
    Query the application mapper for given user agent strings and return
    a UserAgentMapper object with the results.

    Args:
        user_agents (List[str]): A list of user agents to be mapped.
        db_path (str, optional): The path to the user agent mapping database.
            Defaults to DB_PATH.
        logger (logging.Logger): The logger used for logging progress and errors.
        llm_api_key (str, optional): An API key for LLM usage. Defaults to None.
        llm_selection (str, optional): The LLM to use. Defaults to "Gemini".

    Returns:
        UserAgentMapper: An instance of UserAgentMapper containing the mapping results.

    Raises:
        None
    """
    # Eliminate empty user agents from input
    user_agent_input = [
        user_agent for user_agent in user_agents if not re.match("^$", user_agent)
    ]

    logger.info(
        f"A total of {len(user_agent_input)} user agents were provided to the mapper."
    )

    ds = DataStoreHandler(db_path=db_path)

    mapper = UserAgentMapper(
        query_input=user_agent_input,
        llm_api_key=llm_api_key,
        llm_selection=llm_selection,
        logger=logger,
        datastore=ds,
    )
    mapper.search()

    # Don't save the results here, but return the mapper.
    return mapper
