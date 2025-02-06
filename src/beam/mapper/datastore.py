import logging
from typing import List, Tuple

from pydantic import ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy import Engine
from sqlalchemy_utils import create_database, database_exists
from sqlmodel import Session, SQLModel, create_engine, select

from .data_sources import Application, Mapping, OperatingSystem


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
            app (Application): The Application object to add.

        Returns:
            None
        """
        current_app = self.search_applications(name=app.name)
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
            os (OperatingSystem): The OperatingSystem object to add.

        Returns:
            None
        """
        current_os = self.search_operating_systems(name=os.name)
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
            self.logger.info("Mapping already exists in the database.")
            return
        else:
            # The mapping was NOT found in the database.
            self.logger.info("Adding a new mapping to the database.")
            # Check if the application and operating system exist in the database.
            app = self.search_applications(
                session=session, name=mapping.application.name
            )
            if app:
                mapping.application = app
            os = self.search_operating_systems(
                session=session, name=mapping.operatingsystem.name
            )
            if os:
                mapping.operatingsystem = os
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

    def __initialize_database__(self, db_path: str):
        """Initialize the database."""
        self.database = Database(mapping_database_path=db_path, logger=self.logger)

    def __init__(self, db_path: str, logger: logging.Logger):
        self.query_input: List[str] = []
        self.logger = logger
        self.database: Database = None
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
            mapping = self.database.search_user_agents(session=session, ua=user_agent)
            if mapping:
                hits.append(mapping)
            else:
                misses.append(user_agent)
        return hits, misses

    def save_results(self, session: Session, mappings: List[Mapping]) -> None:
        """
        Save a list of Mapping objects to the database.

        Args:
            input (List[Mapping]): A list of Mapping objects to save.

        Returns:
            None
        """
        for mapping in mappings:
            if isinstance(mapping, Mapping):
                self.database.add_user_agent_mapping(session=session, mapping=mapping)
