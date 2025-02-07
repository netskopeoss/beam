from os import environ
from pathlib import Path


def get_project_root() -> Path:
    """
    Returns the project root path.

    Args:
        None

    Returns:
        Path: The root path of the project.

    Raises:
        None
    """
    cmd = Path(__file__)
    return Path([i for i in cmd.parents if i.as_uri().endswith("src")][0]).parent


PROJECT_DIR = get_project_root()

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIRECTORY = PROJECT_DIR / "models"
PREDICTIONS_DIRECTORY = PROJECT_DIR / "predictions"

LOG_CONFIG = PROJECT_DIR / "src" / "beam" / "logging.conf"
DB_PATH = DATA_DIR / "mapper" / "user_agent_mapping.db"
KEY_DOMAINS_FILE = DATA_DIR / "key_domains.json"
CLOUD_DOMAINS_FILE = DATA_DIR / "cloud_domains.json"

ZEEK_OUTPUT_PATH = DATA_DIR / "input_parsed"
ENRICHED_EVENTS_PATH = DATA_DIR / "enriched_events"
FEATURE_SUMMARY_OUTPUT_PATH = DATA_DIR / "summaries"

COMBINED_APP_MODEL = MODEL_DIRECTORY / "supply_chain_combined_app_model.pkl"
INDIVIDUAL_APP_MODEL = MODEL_DIRECTORY / "supply_chain_individual_app_models.pkl"

COMBINED_APP_PREDICTIONS_DIR = PREDICTIONS_DIRECTORY / "anomalous_app"
APP_PREDICTIONS_DIR = PREDICTIONS_DIRECTORY / "anomalous_domains"

GEMINI_API_KEY = environ["GEMINI_API_KEY"] if "GEMINI_API_KEY" in environ else ""
