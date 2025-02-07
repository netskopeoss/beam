import logging.config
from os import path, remove

from beam.detector.utils import get_project_root
from beam.mapper.data_sources import Application, Mapping, OperatingSystem
from beam.mapper.datastore import DataStoreHandler

PROJECT_DIR = get_project_root()
LOG_CONFIG = PROJECT_DIR / "src" / "beam" / "logging.conf"
TEST_DB_PATH = "./test_datastore.db"

logging.config.fileConfig(LOG_CONFIG)
logger = logging.getLogger("test_datastore")

chrome_dict = {
    "user_agent_string": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36",
    "version": "42.0.2311.90",
    "application": {
        "name": "Chrome",
        "vendor": "Google",
        "description": "Chrome is a freeware web browser developed by Google",
    },
    "os": {"name": "Windows 7"},
}


def reset_db():
    """Reset the test database by deleting it."""
    logger.info("Checking for the existence of the datastore test database.")
    if path.exists(TEST_DB_PATH):
        logger.info("Removing the old datastore test database.")
        remove(TEST_DB_PATH)


def test_datastore_save_results():
    """Test the DataStore update methods."""

    reset_db()
    app = Application.model_validate(chrome_dict["application"])
    os = OperatingSystem.model_validate(chrome_dict["os"])
    mapping = Mapping(
        user_agent_string=chrome_dict["user_agent_string"],
        version=chrome_dict["version"],
        application=app,
        operatingsystem=os,
    )
    datastore = DataStoreHandler(db_path=TEST_DB_PATH, logger=logger)
    session = datastore.database.open_database()
    datastore.save_results(session=session, mappings=[mapping])
    session.close()


def test_search_datastore():
    """Test the DataStore search methods."""
    datastore = DataStoreHandler(logger=logger, db_path=TEST_DB_PATH)
    session = datastore.database.open_database()
    hits, misses = datastore.search(
        session=session, user_agents=[chrome_dict["user_agent_string"]]
    )

    print("Here are the hits:")
    print(hits)
    assert len(hits) == 1
    assert len(misses) == 0
