import logging
from typing import List

from pydantic import ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy import Engine
from sqlalchemy_utils import create_database, database_exists
from sqlmodel import Session, SQLModel, create_engine, select

from .data_sources import Application, Mapping, OperatingSystem

logger = logging.getLogger(__name__)


def initialize_database(engine: Engine):
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
    logger: logging.Logger = logger
    db_session: Session = None

    # Needed to allow the logger to be passed in
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def search_applications(self, name: str) -> Application | None:
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
            return self.db_session.exec(statement=query).one()
        except Exception:
            return None

    def search_operating_systems(self, name: str) -> OperatingSystem | None:
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
            return self.db_session.exec(statement=query).one()
        except Exception:
            return None

    def search_user_agents(self, ua: str) -> Mapping | None:
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
            result = self.db_session.exec(query).one_or_none()
        except Exception as e:
            self.logger.error(f"Error querying user agent {ua}: {e}")
        return result

    def add_application(self, app: Application) -> None:
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
            self.db_session.merge(app)
            self.db_session.commit()

    def add_operating_system(self, os: OperatingSystem) -> None:
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
            self.db_session.merge(os)
            self.db_session.commit()

    def add_user_agent_mapping(self, mapping: Mapping) -> None:
        """
        Add a user agent Mapping to the database if it does not already exist.

        Args:
            mapping (Mapping): The Mapping object containing user agent data,
            including an associated Application and OperatingSystem.

        Returns:
            None
        """
        current_mapping = self.search_user_agents(ua=mapping.user_agent_string)
        if current_mapping:
            # The mapping was found in the database.
            self.logger.info("Mapping already exists in the database.")
            return
        else:
            # The mapping was NOT found in the database.
            logger.info("Adding a new mapping to the database.")
            # Check if the application and operating system exist in the database.
            app = self.search_applications(name=mapping.application.name)
            if app:
                mapping.application = app
            os = self.search_operating_systems(mapping.operatingsystem.name)
            if os:
                mapping.operatingsystem = os
            self.db_session.merge(mapping)
            self.db_session.commit()

    def open_database(self) -> None:
        """
        Connect to the database and create a Session object if the database
        does not exist. If it does not exist, it is created.

        Args:
            None

        Returns:
            None
        """
        sqlite_string = "sqlite:///" + self.mapping_database_path
        self.logger.info(f"Connecting to this database: {sqlite_string}")
        # engine = create_engine(sqlite_string, echo=True)
        engine = create_engine(sqlite_string)
        if not database_exists(engine.url):
            self.logger.info("Database does not exist. Creating it now.")
            initialize_database(engine)

        self.db_session = Session(engine)

    def close_database(self) -> None:
        """
        Close the database connection.

        Args:
            None

        Returns:
            None
        """
        # TODO: We are closing the db too early here since lazy loaded mapping fields like application
        #  are not fully loaded yet
        # self.db_session.close()
        pass


class DataStoreHandler:
    """Class to search and update the data store of mapped user agents."""

    def __init__(self, db_path: str):
        self.hits: List[Mapping] = []
        self.misses: List[str] = []
        self.query_input: List[str] = []
        self.logger: logging.Logger = logger
        self.database: Database = Database(
            mapping_database_path=db_path, logger=self.logger
        )

    def hits_found(self) -> bool:
        """
        Check if any hits were found during the last search.

        Args:
            None

        Returns:
            bool: True if hits are present, otherwise False.
        """
        return len(self.hits) > 0

    def misses_found(self) -> bool:
        """
        Check if any misses were found during the last search.

        Args:
            None

        Returns:
            bool: True if misses are present, otherwise False.
        """
        return len(self.misses) > 0

    def search(self, user_agents: List[str]) -> None:
        """
        For each user agent in the provided list, query the database
        and store the results in hits or misses.

        Args:
            user_agents (List[str]): A list of user agent strings to search for.

        Returns:
            None
        """
        self.query_input = user_agents
        self.database.open_database()
        for user_agent in self.query_input:
            result = self.database.search_user_agents(user_agent)
            if result:
                self.hits.append(result)
            else:
                self.misses.append(user_agent)
        self.database.close_database()

    def save_results(self, result_set: List[Mapping]) -> None:
        """
        Save a list of Mapping objects to the database.

        Args:
            result_set (List[Mapping]): A list of Mapping objects to save.

        Returns:
            None
        """
        self.database.open_database()
        for item in result_set:
            if isinstance(item, Mapping):
                self.database.add_user_agent_mapping(item)
        self.database.close_database()
        self.hits.extend(result_set)
