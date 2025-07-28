"""Mapper Module
This module contains the UserAgentMapper class and the
query_user_agent_mapper function.
"""

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

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import ConfigDict
from sqlalchemy.orm import Session

from beam.constants import GEMINI_API_KEY, LLAMA_BASE_URL, USE_LOCAL_LLM
from beam.mapper.data_sources import APIDataSource, DataSource

from .agent_parser import query_agent_parser
from .datastore import DataStoreHandler
from .gemini import query_gemini
from .llama import query_llama
from .llm import LLMDataSource


def mass_mapping(user_agents: list, db_path: Path, logger: logging.Logger) -> None:
    """
    Map a list of user agents to applications.

    Args:
        user_agents (list): A list of user agents to map.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        None

    Raises:
        None
    """
    logger.info(f"Mapping {len(user_agents)} user agents...")
    
    # Choose LLM selection based on configuration
    llm_selection = "Llama" if USE_LOCAL_LLM else "Gemini"
    
    hits, misses = query_user_agent_mapper(
        user_agents=user_agents,
        db_path=str(db_path),
        logger=logger,
        llm_api_key=GEMINI_API_KEY,
        llm_selection=llm_selection,
        delay=1,
    )
    logger.info(f"Found {len(hits)} hits and {len(misses)} misses.")


def run_mapping_only(
    user_agent_file: Path, db_path: Path, chunk_size: int, logger: logging.Logger
) -> None:
    """
    Run the mass mapping process to map user agents to applications.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    logger.info(f"Reading user agents from {user_agent_file}")

    with open(user_agent_file, "r", encoding="utf-8") as file:
        user_agents = file.read().splitlines()

    logger.info(f"Read {len(user_agents)} user agents from the file.")

    for i in range(0, len(user_agents), chunk_size):
        logger.info(f"Processing chunk starting at {i}")
        time.sleep(2)

        # Extract a slice of up to chunk_size items
        chunk = user_agents[i : i + chunk_size]
        # Call the function on this chunk
        mass_mapping(user_agents=chunk, db_path=db_path, logger=logger)

    logger.info("Finished the mapping process.")


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
    llm: LLMDataSource | None = None
    api: APIDataSource | None = None

    # Needed to allow the logger to be passed in
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def search(self, session: Session) -> None:
        """
        Search the data sources in order (datastore, LLM, API) for the user
        agents.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        if len(self.query_input) > 0:
            with session:
                self.search_datastore(session=session)
                self.search_llm()
                self.search_api()

    def search_datastore(self, session: Session) -> None:
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
        hits, misses = self.datastore.search(
            session=session, user_agents=self.query_input
        )
        # Update the hits and misses lists
        if len(hits) > 0:
            self.logger.info(f"Found {len(hits)} user agents in the datastore.")
            self.hits.extend(hits)
        elif len(hits) == 0:
            self.logger.info("No user agents were found in the datastore.")

        if len(misses) > 0:
            self.logger.info(
                f"Found {len(misses)} user agents missing from the datastore."
            )
            self.misses.extend(misses)
        else:
            self.logger.info("Nothing left to search after querying the datastore.")

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
        if self.misses_found:
            self.logger.info(
                f"Checking LLM source {self.llm_selection} for a total of {len(self.misses)} user agents."
            )
            if self.llm_selection == "Llama":
                self.llm = query_llama(
                    user_agents=self.misses,
                    logger=self.logger,
                )
            elif self.llm_api_key and self.llm_selection == "Gemini":
                self.llm = query_gemini(
                    user_agents=self.misses,
                    api_key=self.llm_api_key,
                    logger=self.logger,
                )
            else:
                if self.llm_selection == "Gemini":
                    self.logger.info("No LLM API key was provided for Gemini, skipping LLM check.")
                else:
                    self.logger.info("Unknown LLM selection, skipping LLM check.")
                return

            if self.llm.hits_found:
                self.logger.info(
                    f"{self.llm_selection} was able to map {len(self.llm.hits)} user agents."
                )
                self.hits.extend(self.llm.hits)
            else:
                self.logger.info(
                    f"{self.llm_selection} was unable to map any user agents."
                )

            if self.llm.misses_found:
                # We need to reset the misses list to the new misses
                # Some of the old misses may be hits now
                self.misses = self.llm.misses
            else:
                self.logger.info("Nothing left to search after querying the LLM.")
                return
            self.logger.info(
                f"A total of {len(self.misses)} misses exist after checking {self.llm_selection}."
            )
        else:
            self.logger.info("Skipping the LLM check because 0 misses are left.")

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
        if self.misses_found:
            self.logger.info(
                f"Checking the API for a total of {len(self.misses)} user agents."
            )
            self.api = query_agent_parser(self.misses)
            if self.api.hits_found:
                self.hits.extend(self.api.hits)

            # We need to reset the misses list to the new misses
            # Some of the old misses may be hits now
            self.misses = self.api.misses
            if self.misses_found:
                self.logger.info(
                    f"A total of {len(self.misses)} misses exist after checking the API."
                )
            else:
                self.logger.info(
                    "All user agents were mapped after checking the API data source."
                )
        else:
            self.logger.info("Skipping the API check because 0 misses are left.")

    def save_results(self, session: Session) -> None:
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
            self.datastore.save_results(session=session, mappings=self.hits)
        if self.misses_found:
            self.logger.info(f"The mapper missed {len(self.misses)} user agents.")


def query_user_agent_mapper(
    user_agents: List[str],
    db_path: str,
    logger: logging.Logger,
    llm_api_key: str = None,
    llm_selection: str = "Gemini",
    delay: int = 0,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Query the application mapper for given user agent strings and return the results.

    Args:
        user_agents (List[str]): A list of user agents to be mapped.
        db_path (str, optional): The path to the user agent mapping database.
            Defaults to DB_PATH.
        logger (logging.Logger): The logger used for logging progress and errors.
        llm_api_key (str, optional): An API key for LLM usage. Defaults to None.
        llm_selection (str, optional): The LLM to use. Defaults to "Gemini".

    Returns:
        Hits: Dictionary of mapped user agents
        Misses: List of user agents that were not mapped successfully.

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

    ds = DataStoreHandler(db_path=db_path, logger=logger)
    mapper = UserAgentMapper(
        query_input=user_agent_input,
        llm_api_key=llm_api_key,
        llm_selection=llm_selection,
        logger=logger,
        datastore=ds,
    )
    session = mapper.datastore.database.open_database()

    mapper.search(session=session)
    mapper.save_results(session=session)
    time.sleep(delay)
    hits = []
    if mapper.hits_found:
        hits = []
        # Use no_autoflush to prevent SQLAlchemy warnings when accessing relationships
        with session.no_autoflush:
            for hit in mapper.hits:
                if hit not in session:
                    session.add(hit)

                app_name = hit.application.name
                name_map = {
                    "Asana": [
                        "Asana", "Asana Desktop", "Asana Desktop Official",
                        "Asana Mobile App", "Asana Mobile App", "Asana App"
                    ],
                    "Kandji": [
                        "Kandji Daemon", "Kandji Self Service", "Kandji Menu",
                        "Kandji Library Manager"
                    ],
                    "Todoist": [
                        "Todoist", "TodoistWidgets"
                    ],
                    "Canva": [
                        "Canva", "Canva Editor", "Canva editor"
                    ],
                }

                # If the assignment is one of the values in the array,
                # substitute it with the value in the key
                for k in name_map:
                    if app_name in name_map[k]:
                        app_name = k
                        break

                hits.append(
                    {
                        "user_agent_string": hit.user_agent_string,
                        "application": app_name,
                        "vendor": hit.application.vendor,
                        "version": hit.version,
                        "description": hit.application.description,
                        "operating_system": hit.operatingsystem.name,
                    }
                )
    session.close()
    return hits, mapper.misses
