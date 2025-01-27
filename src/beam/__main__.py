import logging
import logging.config
from os import path
from beam.utils import get_project_root
from beam.detect import detect_anomalous_app
from beam import enrich, features, zeek
from typing import Tuple

# Module constants
PROJECT_DIR = get_project_root()
DATA_DIR = PROJECT_DIR / "data"
LOG_CONFIG = PROJECT_DIR / 'src' / 'beam' / 'logging.conf'

def run_zeek(
        file_path: str,
        logger: logging.Logger
        ) -> Tuple[str, str]:
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
    logger.info(f"Running Zeek on file: {file_path}")
    file_name = file_path.split('/')[-1].replace('.pcap', '')
    
    zeek_path = zeek.run_zeek(file_path, logger)    
    zeek_output_path = f"{DATA_DIR}/zeek_parsed/{file_name}.json"
    zeek.process_zeek_output(
        input_path=zeek_path,
        output_path=zeek_output_path,
        logger=logger
    )
    logger.info(f"Zeek output saved to: {zeek_output_path}")
    return file_name, zeek_output_path

def enrich_zeek_output(
        file_name: str,
        zeek_output_path,
        logger: logging.Logger
        ) -> str:
    """
    Enrich Zeek output and save the enriched data to a new JSON file.

    Args:
        file_name (str): The identifier or name for the PCAP file.
        zeek_output_path (str): The path to the initial Zeek output JSON file.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        str: The path to the newly enriched JSON file.

    Raises:
        None
    """
    enriched_events_path = f"{DATA_DIR}/enriched_events/{file_name}.json"
    enrich.enrich_events(
        input_path=zeek_output_path,
        output_path=enriched_events_path,
        logger=logger
    )
    logger.info(f"Enriched events saved to: {enriched_events_path}")
    return enriched_events_path
    
def run_detection(file_name, enriched_events_path, logger) -> None:
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
    features_output_path = f"{DATA_DIR}/summaries/{file_name}.json"
    # TODO: Make this app field configurable
    features.aggregate_app_traffic(app_field='useragent',
                                   input_path=enriched_events_path,
                                   output_path=features_output_path)
    logger.info(f"Features output saved to: {features_output_path}")

    detect_anomalous_app(
        input_path=features_output_path,
        logger=logger
        )
    logger.info(f"Anomalous app detection completed for: {features_output_path}")

def process_pcap_files(logger: logging.Logger) -> None:
    """
    Process PCAP files made available in the 'input_pcaps' directory, running
    Zeek, enrichment, and detection steps in sequence.

    Args:
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        None

    Raises:
        None
    """
    pcaps_directory = DATA_DIR / "input_pcaps"
    input_pcap_paths = [
        f'{pcaps_directory}/EK_MALWARE_2014-08-14-Fiesta-EK-traffic_mailware-traffic-analysis.net.pcap',
        f'{pcaps_directory}/pcap_pcap_eldracote_CTU-Malware-Capture-Botnet-100_2014-12-20_capture-win5.pcap',
        f'{pcaps_directory}/pcap_pcap_eldracote_CTU-Malware-Capture-Botnet-110-4_2015-04-22_capture-win9.pcap',
        f'{pcaps_directory}/pcap_pcap_eldracote_CTU-Malware-Capture-Botnet-112-2_2015-04-09_capture-win11.first200000.pcap',
        f'{pcaps_directory}/pcap_pcap_eldracote_Android-Mischief-Dataset_AndroidMischiefDataset_v1_RAT03-HawkShaw_RAT03',
        f'{pcaps_directory}/slv-http-20240428-011955_20240428-012530_inner.pcap',
        f'{pcaps_directory}/slv-http-20240428-012202_20240428-052701_inner.pcap',
        f'{pcaps_directory}/slv-http-20240428-012328_20240428-092830_inner.pcap',
        f'{pcaps_directory}/trial_run_2_pcaps_100_26_67_139_1629213298.pcap',
        f'{pcaps_directory}/cobaltstrike_pcaps_cobalt-http-20240414-150357_20240414-190727_inner.pcap',
        f'{pcaps_directory}/toolsmith.pcap'
    ]

    for pcap_file in input_pcap_paths:
        if path.exists(pcap_file):
            logger.info(f"Processing pcap file: {pcap_file}")
            file_name, zeek_output = run_zeek(
                file_path=pcap_file,
                logger=logger
                )
            enriched_events_path = enrich_zeek_output(
                file_name=file_name,
                zeek_output_path=zeek_output,
                logger=logger
                )
            run_detection(
                file_name=file_name,
                enriched_events_path=enriched_events_path,
                logger=logger
                )
        else:
            logger.error(f"File not found: {pcap_file}")

def main():
    """
    Main entry point of beam.

    Initializes logging, processes PCAP files, and orchestrates the entire pipeline.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    logging.config.fileConfig(LOG_CONFIG)
    logger = logging.getLogger("beam.main")

    logger.info("[x] Starting beam processing on PCAP files")
    process_pcap_files(logger=logger)

if __name__ == "__main__":
    main()
