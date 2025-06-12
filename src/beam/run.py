"""Run module for execution"""

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

import argparse
import glob
import json
import logging.config
import warnings
from os import path
from pathlib import Path
from typing import Optional, Tuple

from art import tprint

from beam import constants, enrich
from beam.detector import features, utils
from beam.detector.detect import MultiHotEncoder, detect_anomalous_domain
from beam.detector.trainer import (
    ModelTrainer,
    extract_app_features,
    train_custom_app_model,
)
from beam.mapper.mapper import run_mapping_only
from beam.parser import har, zeek

warnings.filterwarnings(action="ignore")

DATA_DIR = constants.DATA_DIR


def run_detection(
    file_name: str,
    enriched_events_path: str,
    logger: logging.Logger,
    use_custom_models: bool = True,
) -> None:
    """
    Detect anomalous apps in the enriched events by aggregating app traffic
    and applying an anomaly detection method.

    Args:
        file_name (str): The identifier or name for the PCAP file.
        enriched_events_path (str): Path to the enriched events JSON file.
        logger (logging.Logger): Logger instance for capturing log messages.
        use_custom_models (bool): Whether to include custom trained models in detection.

    Returns:
        None

    Raises:
        None
    """
    # Run app detection with basic and custom models
    logger.info("Analysing applications...")
    app_features_output_path = f"{DATA_DIR}/app_summaries/{file_name}.json"
    features.aggregate_app_traffic(
        fields=["useragent"],
        input_path=enriched_events_path,
        output_path=app_features_output_path,
        min_transactions=constants.MIN_APP_TRANSACTIONS,
    )

    logger.info("Analysing domains...")
    features_output_path = f"{DATA_DIR}/domain_summaries/{file_name}.json"
    features.aggregate_app_traffic(
        fields=["application", "domain"],
        input_path=enriched_events_path,
        output_path=features_output_path,
        min_transactions=constants.MIN_DOMAIN_TRANSACTION,
    )
    detect_anomalous_domain(
        input_path=features_output_path,
        domain_model_path=str(constants.DOMAIN_MODEL),
        app_prediction_dir=str(constants.DOMAIN_PREDICTIONS_DIR),
    )
    logger.info(f"Features output saved to: {features_output_path}")


def enrich_events(file_name: str, parsed_file_path, logger: logging.Logger) -> str:
    """
    Enrich Zeek output and save the enriched data to a new JSON file.

    Args:
        file_name (str): The identifier or name for the PCAP file.
        parsed_file_path (str): The path to the initial Zeek output JSON file.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        str: The path to the newly enriched JSON file.

    Raises:
        None
    """
    events = enrich.enrich_events(
        input_path=parsed_file_path,
        db_path=str(constants.DB_PATH),
        cloud_domains_file_path=str(constants.CLOUD_DOMAINS_FILE),
        key_domains_file_path=str(constants.KEY_DOMAINS_FILE),
        llm_api_key=constants.GEMINI_API_KEY,
    )
    enriched_events_path = f"{DATA_DIR}/enriched_events/{file_name}.json"
    utils.save_json_data(events, enriched_events_path)
    logger.info(f"Enriched events saved to: {enriched_events_path}")
    return enriched_events_path


def parse_har(file_path: str, logger: logging.Logger) -> Tuple[str, str]:
    """
    Parse a HAR file and save the output to a JSON file.

    Args:
        file_path (Path): The path to the HAR file to be processed.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        Tuple[str, str]: A tuple containing the file name and the path to the processed HAR output JSON file.

    Raises:
        None
    """
    logger.info(f"Processing har file: {file_path}")
    file_name = file_path.split("/")[-1].replace(".har", "")
    har_output_path = f"{DATA_DIR}/input_parsed/{file_name}.json"
    parsed_responses = har.parse_har_log(file_path)
    result = [json.loads(response.model_dump_json()) for response in parsed_responses]
    utils.save_json_data(result, har_output_path)
    logger.info(f"Processed har output saved to: {har_output_path}")
    return file_name, har_output_path


def parse_pcap(file_path: str, logger: logging.Logger) -> Tuple[str, str]:
    """
    Use Zeek to process a pcap file and save the output to a JSON file.

    Args:
        file_path (str): The path to the PCAP file to process.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        Tuple[str, str]: A tuple containing file_name and the path to the Zeek output.

    Raises:
        None
    """
    logger.info(f"Processing pcap file: {file_path}")
    file_name = file_path.split("/")[-1].replace(".pcap", "").replace(".cap", "")
    zeek_path = zeek.run_zeek(file_path)
    zeek_output_path = f"{DATA_DIR}/input_parsed/{file_name}.json"
    zeek_results = zeek.process_zeek_output(input_path=zeek_path)
    utils.save_json_data(zeek_results, zeek_output_path)
    logger.info(f"Zeek output saved to: {zeek_output_path}")
    return file_name, zeek_output_path


def parse_input_file(file_path: str, logger: logging.Logger) -> Tuple[str, str]:
    """
    Processes the input network file.
    Currently, supports HAR and PCAP files.

    Args:
        file_path (Path): The path to the input network file to be processed.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        Tuple[str, str]: A tuple containing the file name and the path to the processed output JSON file.

    Raises:
        Exception: If the file type is not supported.
    """
    if ".har" in file_path:
        return parse_har(file_path=file_path, logger=logger)
    elif (".pcap" in file_path) or (".cap" in file_path):
        return parse_pcap(file_path=file_path, logger=logger)
    else:
        raise ValueError("[!!] File type is not supported")


def process_input_file(
    file_path: str, logger: logging.Logger, use_custom_models: bool = True
) -> None:
    """
    Process files made available in the 'input_pcaps' directory, running
    Zeek, enrichment, and detection steps in sequence.

    Args:
        file_path (str): Path to the input file to process.
        logger (logging.Logger): Logger instance for capturing log messages.
        use_custom_models (bool): Whether to include custom trained models in detection.

    Returns:
        None

    Raises:
        None
    """
    if path.exists(file_path):
        logger.info(f"Processing file: {file_path}")
        file_name, parsed_file_path = parse_input_file(
            file_path=file_path, logger=logger
        )
        enriched_events_path = enrich_events(
            file_name=file_name, parsed_file_path=parsed_file_path, logger=logger
        )
        run_detection(
            file_name=file_name,
            enriched_events_path=enriched_events_path,
            logger=logger,
            use_custom_models=use_custom_models,
        )
    else:
        logger.error(f"File not found: {file_path}")


def process_training_data(
    input_file_path: str,
    app_name: str,
    custom_model_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Orchestrates the full pipeline for training a custom application model from network traffic data (PCAP or HAR).

    This function performs the following steps:
      1. Ensures a logger is available for status and error reporting.
      2. Determines the output path for the custom model, creating the directory if needed.
      3. Parses the input file (PCAP or HAR) into a standardized JSON format.
      4. Enriches the parsed events with additional context (e.g., domain and cloud data).
      5. Extracts relevant features (e.g., useragent) from the enriched events for model training.
      6. Trains a custom application model using the extracted features and saves it.
      7. If a standard application model exists, merges the new custom model with it to create a combined model, enabling detection of both standard and custom apps.

    This pipeline automates and standardizes the process of training new application detection models, ensuring consistency and enabling seamless extension of the detection system with custom-trained models.

    Args:
        input_file_path (str): Path to the input file (pcap or har).
        app_name (str): Name of the app to train for.
        custom_model_path (Optional[str]): Path to save the model, uses default if None.
        logger (Optional[logging.Logger]): Logger instance.

    Returns:
        None

    Raises:
        None
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not custom_model_path:
        # Ensure custom models directory exists
        safe_create_path = utils.safe_create_path
        safe_create_path(str(constants.CUSTOM_APP_MODELS_DIR))
        custom_model_path = str(
            constants.CUSTOM_APP_MODELS_DIR
            / f"{app_name.replace(' ', '_').lower()}_model.pkl"
        )

    logger.info("Processing training data for app: %s", app_name)

    # Parse the input file
    file_name, parsed_file_path = parse_input_file(
        file_path=input_file_path, logger=logger
    )

    # Enrich the events
    enriched_events_path = enrich_events(
        file_name=file_name, parsed_file_path=parsed_file_path, logger=logger
    )

    # Extract features for model training
    features_output_path = f"{DATA_DIR}/app_summaries/{file_name}.json"
    extract_app_features(
        input_data_path=enriched_events_path,
        output_path=features_output_path,
        min_transactions=constants.MIN_APP_TRANSACTIONS,
        fields=["useragent", "domain"],
    )

    # Train the custom app model with a more reasonable default n_features
    train_custom_app_model(
        features_path=features_output_path,
        app_name=app_name,
        output_model_path=custom_model_path,
        n_features=50,  # Lower default value to avoid exceeding available features
        min_transactions=constants.MIN_APP_TRANSACTIONS,
    )

    logger.info(
        "Custom model for '%s' has been created at: %s", app_name, custom_model_path
    )

    # Create combined model including the new app
    if Path(str(constants.APP_MODEL)).exists():
        combined_model_path = str(constants.MODEL_DIRECTORY / "combined_app_model.pkl")
        trainer = ModelTrainer()
        trainer.merge_models(
            existing_model_path=str(constants.APP_MODEL),
            new_model_path=custom_model_path,
            output_path=combined_model_path,
        )
        logger.info(
            "Created combined model with the new app at: %s", combined_model_path
        )


def run(logger: logging.Logger) -> None:
    """
    Run beam to find anomalous applications

    Args:
        logger

    Returns:
        None

    Raises:
        None
    """

    _m = MultiHotEncoder
    tprint("BEAM", "rand")
    parser = argparse.ArgumentParser(description="BEAM")

    parser.add_argument(
        "-i",
        "--input_dir",
        help="Directory containing the input files to be processed",
        required=False,
        default=DATA_DIR / "input",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="The log level to use for logging",
        required=False,
        default="INFO",
    )
    parser.add_argument(
        "-m",
        "--mapping_only",
        help="Path to an input file of user agents to do mapping only.",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--train",
        help="Train a custom app model using the provided input file.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--app_name",
        help="Name of the application to train the model for. Required with --train.",
        required=False,
    )
    parser.add_argument(
        "--model_output",
        help="Path to save the trained model. Optional with --train.",
        required=False,
    )
    parser.add_argument(
        "--use_custom_models",
        help="Whether to include custom trained models in detection.",
        required=False,
        action="store_true",
        default=True,
    )

    args = vars(parser.parse_args())
    logger.setLevel(args["log_level"])

    if args["mapping_only"]:
        logger.info("Running mapping only...")
        user_agent_file = Path(args["mapping_only"])
        chunk_size = 200
        run_mapping_only(
            user_agent_file=user_agent_file,
            db_path=constants.DB_PATH,
            chunk_size=chunk_size,
            logger=logger,
        )
        return
    elif args["train"]:
        if not args["app_name"]:
            logger.error("Error: --app_name is required when using --train")
            return

        logger.info(f"Running BEAM in training mode for app: {args['app_name']}")
        # Convert input_dir to Path object if it's a string
        input_dir = (
            Path(args["input_dir"])
            if isinstance(args["input_dir"], str)
            else args["input_dir"]
        )
        input_paths = glob.glob(str(input_dir / "*"))

        if not input_paths:
            logger.error(f"No input files found in {args['input_dir']}")
            return

        # Use the first input file for training
        input_path = input_paths[0]
        process_training_data(
            input_file_path=input_path,
            app_name=args["app_name"],
            custom_model_path=args["model_output"],
            logger=logger,
        )
    else:
        logger.info("Running BEAM in detection mode...")
        use_custom_models = args["use_custom_models"]
        logger.info(
            f"Custom models will be {'used' if use_custom_models else 'ignored'} during detection"
        )

        # Convert input_dir to Path object if it's a string
        input_dir = (
            Path(args["input_dir"])
            if isinstance(args["input_dir"], str)
            else args["input_dir"]
        )
        input_paths = glob.glob(str(input_dir / "*"))
        for input_path in input_paths:
            process_input_file(
                file_path=input_path, logger=logger, use_custom_models=use_custom_models
            )
