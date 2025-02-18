"""Constants module"""

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
