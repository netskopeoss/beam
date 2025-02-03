import logging.config
from os import (
    path,
    environ,
    remove
    )
from beam.mapper.mapper import query_user_agent_mapper
from beam.detector.utils import get_project_root

PROJECT_DIR = get_project_root()
LOG_CONFIG = PROJECT_DIR / 'src' / 'beam' / 'logging.conf'
TEST_DB_PATH = "./test_mapper.db"

logging.config.fileConfig(LOG_CONFIG)
logger = logging.getLogger("test_mapper")


def reset_db():
    """Reset the test database by deleting it.
    """
    logger.info("Checking for the existence of the mapper test database.")
    if path.exists(TEST_DB_PATH):
        logger.info("Removing the old mapper test database.")
        remove(TEST_DB_PATH)


def test_mapper() -> None:
    """Test the mapper with a few user agents.
    This includes testing basic LLM functionality.
    """
    reset_db()
    user_agents = ['Mozilla/5.0', 'Safari', 'poser_invalid_browser']
    mapper = query_user_agent_mapper(
        db_path=TEST_DB_PATH,
        user_agents=user_agents,
        llm_api_key=environ['GEMINI_API_KEY'] if 'GEMINI_API_KEY' in environ else '',
        logger=logger
        )
    mapper.save_results()

    assert mapper.hits_found
    print("Here are the misses left:")
    print(mapper.misses)

