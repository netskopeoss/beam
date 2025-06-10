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
@mock.patch("beam.run.extract_app_features")
@mock.patch("beam.run.train_custom_app_model")
@mock.patch("beam.run.ModelTrainer")
def test_process_training_data_merges_model(
    mock_trainer, mock_train, mock_extract, mock_enrich, mock_parse
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
    mock_extract.return_value = None
    mock_train.return_value = None
    # Simulate APP_MODEL exists
    with mock.patch("beam.run.Path.exists", return_value=True):
        trainer_instance = mock_trainer.return_value
        run.process_training_data(
            input_file_path=input_file,
            app_name=app_name,
            custom_model_path=custom_model_path,
            logger=logger,
        )
        # Check merge_models called
        assert trainer_instance.merge_models.called


@mock.patch("beam.run.parse_input_file")
@mock.patch("beam.run.enrich_events")
@mock.patch("beam.run.extract_app_features")
@mock.patch("beam.run.train_custom_app_model")
def test_process_training_data_no_merge(
    mock_train, mock_extract, mock_enrich, mock_parse
):
    logger = make_logger()
    input_file = "dummy_input.pcap"
    app_name = "TestApp"
    custom_model_path = "models/custom_models/testapp_model.pkl"
    file_name = "dummy_input"
    parsed_file_path = "parsed.json"
    enriched_events_path = "enriched.json"

    mock_parse.return_value = (file_name, parsed_file_path)
    mock_enrich.return_value = enriched_events_path
    mock_extract.return_value = None
    mock_train.return_value = None
    # Simulate APP_MODEL does not exist
    with mock.patch("beam.run.Path.exists", return_value=False):
        run.process_training_data(
            input_file_path=input_file,
            app_name=app_name,
            custom_model_path=custom_model_path,
            logger=logger,
        )
    # No exception means pass


# Optionally, add more tests for error handling, default path creation, etc.
