import logging
import logging.config
from os import (
    path,
    remove
    )
from beam.mapper.datastore import DataStoreHandler
from beam.mapper.data_sources import Application, OperatingSystem, Mapping
from beam.utils import get_project_root

PROJECT_DIR = get_project_root()
LOG_CONFIG = PROJECT_DIR / 'src' / 'beam' / 'logging.conf'
TEST_DB_PATH = "./test_datastore.db"

logging.config.fileConfig(LOG_CONFIG)
logger = logging.getLogger("test_datastore")

def reset_db():
    """Reset the test database by deleting it.
    """
    logger.info("Checking for the existence of the datastore test database.")
    if path.exists(TEST_DB_PATH):
        logger.info("Removing the old datastore test database.")
        remove(TEST_DB_PATH)

def test_datastore_update():
    reset_db()
    chrome_dict = {
        'user_agent_string': "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36",
        'version': "42.0.2311.90",
        'application': {
            'name': "Chrome",
            'vendor': "Google",
            'description': "Chrome is a freeware web browser developed by Google"
        },
        'os': {
            'name': "Windows 7"
        }
    }
    app = Application.model_validate(chrome_dict['application'])
    os = OperatingSystem.model_validate(chrome_dict['os'])
    ua = Mapping(
        user_agent_string=chrome_dict['user_agent_string'],
        version=chrome_dict['version'],
        application=app,
        operatingsystem=os
        )
    datastore = DataStoreHandler(db_path=TEST_DB_PATH)
    datastore.save_results([ua])
    assert datastore.hits_found()