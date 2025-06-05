"""Datastore module"""

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
from typing import List, Optional, Tuple

from pydantic import ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy import Engine
from sqlalchemy_utils import create_database, database_exists
from sqlmodel import Session, SQLModel, create_engine, select

from beam.mapper.data_sources import Application, Mapping, OperatingSystem


def initialize_database(engine: Engine, logger: logging.Logger) -> None:
    """
    Create the database if it doesn't exist, then create all of the tables.

    Args:
        engine (Engine): The SQLAlchemy engine to connect to the database.

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If there is an issue creating the database.
    """
    logger.info("Creating the database.")
    create_database(engine.url)
    # Create all of the tables
    SQLModel.metadata.create_all(engine)


class Database(SQLModel):
    """Class for database operations."""

    mapping_database_path: str = PydanticField()
    logger: logging.Logger

    # Needed to allow the logger to be passed in
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def search_applications(self, session: Session, name: str) -> Application | None:
        """
        Query the database for a particular application name.

        Args:
            session (Session): The database session to use for the query.
            name (str): The name of the application to search for.

        Returns:
            Application | None: The matching Application object if found,
            otherwise None.

        Raises:
            Exception: If an error occurs during the query.
        """
        query = select(Application).where(Application.name == name)
        try:
            return session.exec(statement=query).one()
        except Exception:
            return None

    def search_operating_systems(
        self, session: Session, name: str
    ) -> OperatingSystem | None:
        """
        Query the database for a particular operating system name.

        Args:
            session (Session): The database session to use for the query.
            name (str): The name of the operating system to search for.

        Returns:
            OperatingSystem | None: The matching OperatingSystem object if found,
            otherwise None.

        Raises:
            Exception: If an error occurs during the query.
        """
        query = select(OperatingSystem).where(OperatingSystem.name == name)
        try:
            return session.exec(statement=query).one()
        except Exception:
            return None

    def search_user_agents(self, session: Session, ua: str) -> Mapping | None:
        """
        Query the database for a particular user agent string and return
        a Mapping object with related Application and OperatingSystem objects.

        Args:
            session (Session): The database session to use for the query.
            ua (str): The user agent string to search for.

        Returns:
            Mapping | None: A Mapping object if found, otherwise None.

        Raises:
            Exception: If an error occurs during the query.
        """
        query = select(Mapping).where(Mapping.user_agent_string == ua)
        try:
            mapping = session.exec(query).first()
            if mapping:
                return mapping
        except Exception as e:
            self.logger.error(f"Error querying user agent {ua}: {e}")

    def add_application(self, session: Session, app: Application) -> None:
        """
        Add an Application object to the database if it does not already exist.

        Args:
            session (Session): The database session to use for the query.
            app (Application): The Application object to add.

        Returns:
            None
        """
        current_app = self.search_applications(session=session, name=app.name)
        if current_app:
            self.logger.info(f"Application {current_app} already exists")
            return
        else:
            session.add(app)
            session.commit()

    def add_operating_system(self, session: Session, os: OperatingSystem) -> None:
        """
        Add an OperatingSystem object to the database if it does not already exist.

        Args:
            session (Session): The database session to use for the query.
            os (OperatingSystem): The OperatingSystem object to add.

        Returns:
            None
        """
        current_os = self.search_operating_systems(session=session, name=os.name)
        if current_os:
            self.logger.info(f"Operating System {current_os} already exists")
            return
        else:
            session.add(os)
            session.commit()

    def add_user_agent_mapping(self, session: Session, mapping: Mapping) -> None:
        """
        Add a user agent Mapping to the database if it does not already exist.

        Args:
            mapping (Mapping): The Mapping object containing user agent data,
            including an associated Application and OperatingSystem.

        Returns:
            None
        """
        current_mapping = self.search_user_agents(
            session=session, ua=mapping.user_agent_string
        )
        if current_mapping:
            # The mapping was found in the database.
            return
        else:
            # The mapping was NOT found in the database.
            # Check if the application and operating system exist in the database.
            app = self.search_applications(
                session=session, name=mapping.application.name
            )
            if app:
                mapping.application = app
            os_obj = self.search_operating_systems(
                session=session, name=mapping.operatingsystem.name
            )
            if os_obj:
                mapping.operatingsystem = os_obj
            session.add(mapping)
            session.commit()

    def open_database(self) -> Session:
        """
        Connect to the database and create a Session object if the database
        does not exist. If it does not exist, it is created.

        Args:
            None

        Returns:
            Session: The SQLAlchemy Session object.
        """
        sqlite_string = "sqlite:///" + self.mapping_database_path
        self.logger.info(f"Connecting to this database: {sqlite_string}")
        # engine = create_engine(sqlite_string, echo=True)
        engine = create_engine(sqlite_string)
        if not database_exists(engine.url):
            self.logger.info("Database does not exist. Creating it now.")
            initialize_database(engine=engine, logger=self.logger)

        return Session(engine)


class DataStoreHandler:
    """Class to search and update the data store of mapped user agents."""

    def __initialize_database__(self, db_path: str) -> None:
        """Initialize the database."""
        self.database = Database(mapping_database_path=db_path, logger=self.logger)

    def __init__(self, db_path: str, logger: logging.Logger):
        self.query_input: List[str] = []
        self.logger = logger
        self.database: Optional[Database] = None
        self.__initialize_database__(db_path=db_path)

    def search(
        self, session: Session, user_agents: List[str]
    ) -> Tuple[List[Mapping], List[str]]:
        """
        For each user agent in the provided list, query the database
        and store the results in hits or misses.

        Args:
            user_agents (List[str]): A list of user agent strings to search for.

        Returns:
            Tuple[List[Mapping], List[str]]: A tuple containing the list of
            hits (found user agents) and misses (not found user agents).
        """
        hits: List[Mapping] = []
        misses: List[str] = []

        self.query_input = user_agents
        for user_agent in self.query_input:
            if self.database:
                mapping = self.database.search_user_agents(
                    session=session, ua=user_agent
                )
                if mapping:
                    hits.append(mapping)
                else:
                    misses.append(user_agent)
        return hits, misses

    def save_results(self, session: Session, mappings: List[Mapping]) -> None:
        """
        Save a list of Mapping objects to the database.

        Args:
            session (Session): The database session to use for the query.
            mappings (List[Mapping]): A list of Mapping objects to save.

        Returns:
            None
        """
        for mapping in mappings:
            if isinstance(mapping, Mapping) and self.database:
                self.database.add_user_agent_mapping(session=session, mapping=mapping)
