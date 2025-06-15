import logging
import sys
from unittest import mock

from beam import run


def make_logger():
    logger = logging.getLogger("test_custom_model")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


@mock.patch("beam.run.parse_input_file")
@mock.patch("beam.run.enrich_events")
@mock.patch("beam.run.discover_apps_in_traffic")
@mock.patch("beam.run.extract_app_features")
@mock.patch("beam.run.train_custom_app_model")
def test_process_training_data_with_specific_app(
    mock_train, mock_extract, mock_discover, mock_enrich, mock_parse
):
    # Setup
    logger = make_logger()
    input_file = "dummy_input.pcap"
    app_name = "TestApp"
    custom_model_path = "models/custom_models/testapp_model.pkl"
    file_name = "dummy_input"
    parsed_file_path = "parsed.json"
    enriched_events_path = "enriched.json"

    mock_parse.return_value = (file_name, parsed_file_path)
    mock_enrich.return_value = enriched_events_path
    mock_discover.return_value = {"TestApp": 100}  # Sufficient transactions
    mock_extract.return_value = None
    mock_train.return_value = None

    run.process_training_data(
        input_file_path=input_file,
        app_name=app_name,
        custom_model_path=custom_model_path,
        logger=logger,
    )

    # Check that training was called for the specified app
    mock_train.assert_called_once()


@mock.patch("beam.run.parse_input_file")
@mock.patch("beam.run.enrich_events")
@mock.patch("beam.run.discover_apps_in_traffic")
@mock.patch("beam.run.extract_app_features")
@mock.patch("beam.run.train_custom_app_model")
def test_process_training_data_auto_discovery(
    mock_train, mock_extract, mock_discover, mock_enrich, mock_parse
):
    logger = make_logger()
    input_file = "dummy_input.pcap"
    app_name = None  # No specific app - auto-discovery mode
    custom_model_path = None
    file_name = "dummy_input"
    parsed_file_path = "parsed.json"
    enriched_events_path = "enriched.json"

    mock_parse.return_value = (file_name, parsed_file_path)
    mock_enrich.return_value = enriched_events_path
    # Mock discovering multiple apps - both calls return the same for simplicity
    mock_discover.return_value = {"TestApp": 150, "AnotherApp": 120}
    mock_extract.return_value = None
    mock_train.return_value = None

    run.process_training_data(
        input_file_path=input_file,
        app_name=app_name,
        custom_model_path=custom_model_path,
        logger=logger,
    )

    # Check that training was called (at least once for discovered apps)
    assert mock_train.call_count >= 1  # Should train for discovered apps


# Optionally, add more tests for error handling, default path creation, etc.
