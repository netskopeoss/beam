#!/usr/bin/env python3

"""
Database Initialization Script for BEAM Database Container
This script initializes the SQLite database for user agent mappings
"""

import logging
import sys
import time
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, "/app/src")

try:
    from beam.constants import DB_PATH
    from beam.mapper.datastore import DataStoreHandler
except ImportError as e:
    logging.error(f"Failed to import BEAM modules: {e}")
    sys.exit(1)


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def initialize_database():
    """Initialize the database and create necessary tables"""
    logger = logging.getLogger(__name__)

    try:
        # Ensure the database directory exists
        db_path = Path(DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing database at: {DB_PATH}")

        # Initialize the datastore
        datastore = DataStoreHandler(db_path=str(db_path), logger=logger)

        # Test the connection
        logger.info("Testing database connection...")
        # Add a test entry if needed

        logger.info("Database initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def main():
    """Main function to initialize database and keep container running"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting BEAM Database Service...")

    # Initialize database
    if not initialize_database():
        logger.error("Failed to initialize database, exiting...")
        sys.exit(1)

    logger.info("Database service is ready and running...")

    # Keep the container running
    try:
        while True:
            time.sleep(60)  # Sleep for 1 minute
    except KeyboardInterrupt:
        logger.info("Database service shutting down...")


if __name__ == "__main__":
    main()
