import argparse
import glob
import json
import logging.config
import warnings
from os import path
from parser import har, zeek
from pathlib import Path
from typing import Tuple

import constants
import enrich
from art import tprint
from detector import features, utils
from detector.detect import (
    MultiHotEncoder,
    detect_anomalous_app,
    detect_anomalous_domain,
)
from mapper.mapper import run_mapping_only

warnings.filterwarnings(action="ignore")

DATA_DIR = constants.DATA_DIR


def run_detection(
    file_name: str, enriched_events_path: str, logger: logging.Logger
) -> None:
    """
    Detect anomalous apps in the enriched events by aggregating app traffic
    and applying an anomaly detection method.

    Args:
        file_name (str): The identifier or name for the PCAP file.
        enriched_events_path (str): Path to the enriched events JSON file.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        None

    Raises:
        None
    """
    logger.info("Analysing applications...")
    features_output_path = f"{DATA_DIR}/summaries/{file_name}.json"
    features.aggregate_app_traffic(
        fields=["useragent"],  # TODO: Make this app field configurable
        input_path=enriched_events_path,
        output_path=features_output_path,
    )
    detect_anomalous_app(
        input_path=features_output_path,
        combined_app_model_path=constants.COMBINED_APP_MODEL,
        combined_app_prediction_directory=constants.COMBINED_APP_PREDICTIONS_DIR,
    )

    logger.info("Analysing domains...")
    features_output_path = f"{DATA_DIR}/supply_chain_summaries/{file_name}.json"
    features.aggregate_app_traffic(
        fields=["application", "domain"],
        input_path=enriched_events_path,
        output_path=features_output_path,
    )
    detect_anomalous_domain(
        input_path=features_output_path,
        app_model_path=constants.INDIVIDUAL_APP_MODEL,
        app_prediction_dir=constants.APP_PREDICTIONS_DIR,
    )
    logger.info(f"Features output saved to: {features_output_path}")


def enrich_output(file_name: str, parsed_file_path, logger: logging.Logger) -> str:
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
        cloud_domains_file_path=constants.CLOUD_DOMAINS_FILE,
        key_domains_file_path=constants.KEY_DOMAINS_FILE,
        llm_api_key=constants.GEMINI_API_KEY,
    )
    enriched_events_path = f"{DATA_DIR}/enriched_events/{file_name}.json"
    utils.save_json_data(events, enriched_events_path)
    logger.info(f"Enriched events saved to: {enriched_events_path}")
    return enriched_events_path


def parse_har(file_path: Path, logger: logging.Logger) -> Tuple[str, str]:
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


def parse_pcap(file_path: Path, logger: logging.Logger) -> Tuple[str, str]:
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


def parse_input_file(file_path: Path, logger: logging.Logger) -> Tuple[str, str]:
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
        raise Exception("[!!] File type is not supported")


def process_input_file(file_path: Path, logger: logging.Logger) -> None:
    """
    Process files made available in the 'input_pcaps' directory, running
    Zeek, enrichment, and detection steps in sequence.

    Args:
        file_path:

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
        enriched_events_path = enrich_output(
            file_name=file_name, parsed_file_path=parsed_file_path, logger=logger
        )
        run_detection(
            file_name=file_name,
            enriched_events_path=enriched_events_path,
            logger=logger,
        )
    else:
        logger.error(f"File not found: {file_path}")


def run(logger: logging.Logger) -> None:
    """
    Run beam to find anomalous applications

    Args:
        None

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

    logger.info("Running BEAM...")
    input_paths = glob.glob(str(args["input_dir"] / "*"))
    for input_path in input_paths:
        process_input_file(file_path=input_path, logger=logger)
