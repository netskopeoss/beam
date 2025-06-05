"""Tests for the mapper component."""

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

import logging.config
from os import environ, path, remove

from beam.constants import LOG_CONFIG
from beam.mapper.mapper import query_user_agent_mapper

TEST_DB_PATH = "./test_mapper.db"

logging.config.fileConfig(LOG_CONFIG)
logger = logging.getLogger("test_mapper")


def reset_db() -> None:
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
