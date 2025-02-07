import logging.config
from os import environ, path, remove

from beam.constants import LOG_CONFIG
from beam.mapper.mapper import query_user_agent_mapper

TEST_DB_PATH = "./test_mapper.db"

logging.config.fileConfig(LOG_CONFIG)
logger = logging.getLogger("test_mapper")


def reset_db():
    """Reset the test database by deleting it."""
    logger.info("Checking for the existence of the mapper test database.")
    if path.exists(TEST_DB_PATH):
        logger.info("Removing the old mapper test database.")
        remove(TEST_DB_PATH)


def test_mapper() -> None:
    """Test the mapper with a few user agents.
    This includes testing basic LLM functionality.
    """
    reset_db()
    user_agents = ["Mozilla/5.0", "Safari", "poser_invalid_browser"]
    hits, misses = query_user_agent_mapper(
        db_path=TEST_DB_PATH,
        user_agents=user_agents,
        llm_api_key=environ["GEMINI_API_KEY"] if "GEMINI_API_KEY" in environ else "",
        logger=logger,
    )
    assert len(hits) > 0
    print("Here are the hits:")
    print(hits)
    print("Here are the misses left:")
    print(misses)
