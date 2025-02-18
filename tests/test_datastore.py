"""Tests for the datastore portion of the mapper."""

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
from os import path, remove

from beam.constants import LOG_CONFIG
from beam.mapper.data_sources import Application, Mapping, OperatingSystem
from beam.mapper.datastore import DataStoreHandler

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
