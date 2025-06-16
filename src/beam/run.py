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
from beam.detector.detect import (
    MultiHotEncoder,
    detect_anomalous_domain,
    detect_anomalous_domain_with_custom_model,
)
from beam.detector.security_report import generate_security_report
from beam.detector.trainer import (
    extract_app_features,
    train_custom_app_model,
)
from beam.mapper.mapper import run_mapping_only
from beam.parser import har, zeek

warnings.filterwarnings(action="ignore")

DATA_DIR = constants.DATA_DIR


def normalize_app_name(app_name: str) -> str:
    """
    Normalize application name for consistent file naming.

    Args:
        app_name (str): Original application name

    Returns:
        str: Normalized name (lowercase, spaces replaced with underscores)
    """
    return app_name.lower().replace(" ", "_").replace("-", "_")


def discover_apps_in_traffic(
    enriched_events_path: str, min_transactions: int = 50
) -> dict:
    """
    Discover applications in enriched events and count their transactions.

    Args:
        enriched_events_path (str): Path to enriched events JSON file
        min_transactions (int): Minimum transactions required for an app

    Returns:
        dict: Dictionary mapping original app names to transaction counts
    """
    events = utils.load_json_file(enriched_events_path)
    app_counts = {}

    # Handle cases where events might not be a list
    if not isinstance(events, list):
        raise TypeError(f"Expected events to be a list, got {type(events)}")

    for event in events:
        # Skip malformed events (None, strings, etc.)
        if not isinstance(event, dict):
            continue

        app_name = event.get("application", "Unknown")
        if app_name != "Unknown":
            app_counts[app_name] = app_counts.get(app_name, 0) + 1

    # Filter apps with sufficient transactions
    return {
        app: count for app, count in app_counts.items() if count >= min_transactions
    }


def run_detection(
    file_name: str,
    enriched_events_path: str,
    logger: logging.Logger,
    use_custom_models: bool = False,
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

    if use_custom_models:
        # Discover applications in traffic and match to custom models
        discovered_apps = discover_apps_in_traffic(
            enriched_events_path, min_transactions=constants.MIN_DOMAIN_TRANSACTION
        )

        custom_models_used = False
        for original_app_name in discovered_apps.keys():
            normalized_app_name = normalize_app_name(original_app_name)
            custom_model_path = Path(
                constants.CUSTOM_APP_MODELS_DIR / f"{normalized_app_name}_model.pkl"
            )

            if custom_model_path.exists():
                logger.info(
                    f"Using custom model for '{original_app_name}' -> {custom_model_path}"
                )
                detect_anomalous_domain_with_custom_model(
                    input_path=features_output_path,
                    custom_model_path=custom_model_path,
                    app_prediction_dir=str(constants.DOMAIN_PREDICTIONS_DIR),
                )
                custom_models_used = True
            else:
                logger.info(
                    f"No custom model found for '{original_app_name}' (looked for: {custom_model_path})"
                )

        if not custom_models_used:
            logger.warning(
                "Custom models requested but none found for applications in traffic. Using default domain model."
            )
            model_path = Path(constants.DOMAIN_MODEL)
            detect_anomalous_domain(
                input_path=features_output_path,
                domain_model_path=model_path,
                app_prediction_dir=str(constants.DOMAIN_PREDICTIONS_DIR),
            )
    else:
        model_path = Path(constants.DOMAIN_MODEL)
        logger.info("Using default domain model.")
        detect_anomalous_domain(
            input_path=features_output_path,
            domain_model_path=model_path,
            app_prediction_dir=str(constants.DOMAIN_PREDICTIONS_DIR),
        )
    logger.info(f"Features output saved to: {features_output_path}")
    
    # Generate security analysis report
    try:
        summaries = utils.load_json_file(features_output_path)
        security_report = generate_security_report(summaries)
        
        # Save security report
        security_report_path = features_output_path.replace('.json', '_security_report.txt')
        with open(security_report_path, 'w') as f:
            f.write(security_report)
        
        logger.info(f"Security analysis report saved to: {security_report_path}")
        
        # Also print key security insights to console
        print("\n" + "=" * 60)
        print("🔒 SECURITY ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Extract key insights for console display
        from beam.detector.security_report import SecurityAnalysisReport
        analyzer = SecurityAnalysisReport()
        analysis = analyzer.analyze_security_features(summaries)
        
        # Show critical insights
        critical_insights = [insight for insight in analysis['security_insights'] 
                           if insight['severity'] in ['HIGH', 'MEDIUM']]
        
        if critical_insights:
            print("🚨 CRITICAL SECURITY INSIGHTS:")
            for insight in critical_insights:
                print(f"  [{insight['severity']}] {insight['type']}: {insight['domain']}")
                print(f"      {insight['details'][:80]}...")
        else:
            print("✅ No critical security issues detected")
        
        # Show overall risk
        risk_level = analysis['risk_assessment']['overall_risk_level']
        print(f"\n⚠️  OVERALL RISK LEVEL: {risk_level}")
        print(f"📊 Applications analyzed: {analysis['risk_assessment']['total_applications']}")
        print(f"🔍 Applications with issues: {analysis['risk_assessment']['applications_with_issues']}")
        
        print(f"\n📄 Full report available at: {security_report_path}")
        print("=" * 60)
        
    except Exception as e:
        logger.warning(f"Failed to generate security report: {e}")


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
    file_path: str,
    logger: logging.Logger,
    use_custom_models: bool = False,
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
    app_name: Optional[str] = None,
    custom_model_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Orchestrates the full pipeline for training custom application models from network traffic data (PCAP or HAR).

    This function performs the following steps:
      1. Ensures a logger is available for status and error reporting.
      2. Parses the input file (PCAP or HAR) into a standardized JSON format.
      3. Enriches the parsed events with additional context (e.g., domain and cloud data).
      4. Auto-discovers applications in the traffic (or uses the specified app_name if provided).
      5. For each application with sufficient traffic, extracts relevant features and trains a custom model.
      6. Saves individual models with normalized names for consistent file naming.

    Args:
        input_file_path (str): Path to the input file (pcap or har).
        app_name (Optional[str]): Name of specific app to train for. If None, trains for all discovered apps.
        custom_model_path (Optional[str]): Path to save the model, uses default if None.
        logger (Optional[logging.Logger]): Logger instance.

    Returns:
        None

    Raises:
        None
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Ensure custom models directory exists
    safe_create_path = utils.safe_create_path
    safe_create_path(str(constants.CUSTOM_APP_MODELS_DIR))

    logger.info("Processing training data from: %s", input_file_path)

    # Parse the input file
    file_name, parsed_file_path = parse_input_file(
        file_path=input_file_path, logger=logger
    )

    # Enrich the events
    enriched_events_path = enrich_events(
        file_name=file_name, parsed_file_path=parsed_file_path, logger=logger
    )

    # Discover applications in the traffic
    discovered_apps = discover_apps_in_traffic(
        enriched_events_path, min_transactions=constants.MIN_APP_TRANSACTIONS
    )

    # Report all applications found in traffic (including those below threshold)
    all_apps = discover_apps_in_traffic(enriched_events_path, min_transactions=1)
    logger.info("=== APPLICATION DISCOVERY REPORT ===")
    logger.info("Applications found in traffic:")
    for discovered_app_name, count in sorted(all_apps.items(), key=lambda x: x[1], reverse=True):
        status = (
            "✓ ELIGIBLE"
            if count >= constants.MIN_APP_TRANSACTIONS
            else "✗ insufficient"
        )
        logger.info(f"  {discovered_app_name}: {count} transactions ({status})")
    logger.info(
        f"Minimum transactions required for training: {constants.MIN_APP_TRANSACTIONS}"
    )
    logger.info("=====================================")

    if not discovered_apps:
        logger.warning(
            "No applications found with sufficient transactions for model training"
        )
        return

    logger.info("Applications eligible for training: %s", list(discovered_apps.keys()))

    # Determine which apps to train models for
    if app_name:
        # Check if the specified app exists in the traffic
        if app_name in discovered_apps:
            apps_to_train = {app_name: discovered_apps[app_name]}
            logger.info(
                "Training model for specified app: %s (%d transactions)",
                app_name,
                discovered_apps[app_name],
            )
        else:
            logger.error(
                "Specified app '%s' not found in traffic. Available apps: %s",
                app_name,
                list(discovered_apps.keys()),
            )
            return
    else:
        # Train models for all discovered apps
        apps_to_train = discovered_apps
        logger.info(
            "Training models for all discovered apps: %s", list(apps_to_train.keys())
        )

    # Extract features for model training (once for all apps)
    features_output_path = f"{DATA_DIR}/app_summaries/{file_name}.json"
    # Use both useragent and domain fields for feature extraction to satisfy trainer expectations
    extract_app_features(
        input_data_path=enriched_events_path,
        output_path=features_output_path,
        min_transactions=constants.MIN_APP_TRANSACTIONS,
        fields=["useragent", "domain"],
    )

    # Train models for each app
    for original_app_name, transaction_count in apps_to_train.items():
        normalized_app_name = normalize_app_name(original_app_name)

        if custom_model_path and len(apps_to_train) == 1:
            # Use the provided path if training only one app
            model_path = custom_model_path
        else:
            # Generate path using normalized name
            model_path = str(
                constants.CUSTOM_APP_MODELS_DIR / f"{normalized_app_name}_model.pkl"
            )

        logger.info(
            "Training model for '%s' (%d transactions) -> %s",
            original_app_name,
            transaction_count,
            model_path,
        )

        # Train the custom app model
        train_custom_app_model(
            features_path=features_output_path,
            app_name=original_app_name,  # Use original name for training (it's used as the key)
            output_model_path=model_path,
            n_features=50,  # Lower default value to avoid exceeding available features
            min_transactions=constants.MIN_APP_TRANSACTIONS,
        )

        logger.info(
            "Custom model for '%s' created at: %s", original_app_name, model_path
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
        help="Name of specific application to train model for. If not provided, trains models for all discovered apps with sufficient traffic.",
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
        default=False,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        help="Operation mode: 'demo' to run the supply chain compromise detection demo",
        choices=["demo"],
        default=None,
    )

    args = vars(parser.parse_args())
    logger.setLevel(args["log_level"])

    # Handle demo mode
    if args.get("mode") == "demo":
        logger.info("Running BEAM in demo mode...")
        from beam.demo import run_demo
        run_demo(logger)
        return
    
    # Check if this might be a first-time user with no input files
    input_path_str = args["input_dir"]
    input_path = Path(input_path_str)
    
    # Suggest demo mode for first-time users
    if not args["mapping_only"] and not args["train"]:
        if input_path.is_dir():
            input_files = glob.glob(str(input_path / "*"))
            if not input_files:
                print("\n" + "=" * 60)
                print("👋 WELCOME TO BEAM!")
                print("=" * 60)
                print("No input files found in the data/input directory.")
                print()
                print("🎯 NEW TO BEAM? Try the demo first:")
                print("   python -m beam demo")
                print()
                print("This will showcase BEAM's supply chain compromise detection")
                print("capabilities using real network traffic data.")
                print()
                print("📁 TO ANALYZE YOUR OWN DATA:")
                print("   1. Place .har or .pcap files in data/input/")
                print("   2. Run: python -m beam")
                print("=" * 60)
                return

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
        app_name = args["app_name"]
        if app_name:
            logger.info(f"Running BEAM in training mode for specific app: {app_name}")
        else:
            logger.info("Running BEAM in training mode for all discovered apps")

        # Handle both directory and single file inputs
        input_path_str = args["input_dir"]
        input_path = Path(input_path_str)

        if input_path.is_file():
            # Single file provided
            input_files = [str(input_path)]
        elif input_path.is_dir():
            # Directory provided, find all files
            input_files = glob.glob(str(input_path / "*"))
        else:
            logger.error(f"Input path does not exist: {input_path_str}")
            return

        if not input_files:
            logger.error(f"No input files found in {input_path_str}")
            return

        # Use the first input file for training
        input_file = input_files[0]
        process_training_data(
            input_file_path=input_file,
            app_name=app_name,
            custom_model_path=args["model_output"],
            logger=logger,
        )
    else:
        logger.info("Running BEAM in detection mode...")
        use_custom_models = args["use_custom_models"]
        logger.info(
            f"Custom models will be {'used' if use_custom_models else 'ignored'} during detection"
        )

        # Handle both directory and single file inputs
        input_path_str = args["input_dir"]
        input_path = Path(input_path_str)

        if input_path.is_file():
            # Single file provided
            input_files = [str(input_path)]
        elif input_path.is_dir():
            # Directory provided, find all files
            input_files = glob.glob(str(input_path / "*"))
        else:
            logger.error(f"Input path does not exist: {input_path_str}")
            return

        if not input_files:
            logger.error(f"No input files found in {input_path_str}")
            return

        # Check for matching application models before processing
        if use_custom_models:
            logger.info("Checking for matching application models...")
            
            # Analyze input files to discover applications
            all_discovered_apps = set()
            available_models = set()
            
            # Check what custom models are available
            custom_models_dir = Path(constants.CUSTOM_APP_MODELS_DIR)
            if custom_models_dir.exists():
                for model_file in custom_models_dir.glob("*_model.pkl"):
                    # Extract app name from model filename
                    model_name = model_file.stem.replace("_model", "").replace("_", " ").title()
                    available_models.add(model_name)
            
            # Quick discovery of apps in input files
            for input_file in input_files:
                try:
                    # Parse and enrich a single file to check applications
                    temp_file_name, temp_parsed_path = parse_input_file(
                        file_path=input_file, logger=logger
                    )
                    temp_enriched_path = enrich_events(
                        file_name=temp_file_name, 
                        parsed_file_path=temp_parsed_path, 
                        logger=logger
                    )
                    
                    # Discover applications
                    discovered_apps = discover_apps_in_traffic(
                        temp_enriched_path, min_transactions=1
                    )
                    all_discovered_apps.update(discovered_apps.keys())
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze {input_file} for applications: {e}")
                    continue
            
            if all_discovered_apps:
                logger.info(f"Applications discovered in input data: {sorted(all_discovered_apps)}")
                logger.info(f"Available custom models: {sorted(available_models)}")
                
                # Check for matches (normalize names for comparison)
                normalized_discovered = {normalize_app_name(app) for app in all_discovered_apps}
                normalized_available = {normalize_app_name(model) for model in available_models}
                
                matching_apps = normalized_discovered.intersection(normalized_available)
                missing_apps = normalized_discovered - normalized_available
                
                if not matching_apps:
                    print("\n" + "=" * 60)
                    print("🚫 NO MATCHING MODELS FOUND")
                    print("=" * 60)
                    print("BEAM detected applications in your input data, but no matching")
                    print("trained models were found. You need to train models first.")
                    print()
                    print("🔍 APPLICATIONS DETECTED:")
                    for app in sorted(all_discovered_apps):
                        print(f"   • {app}")
                    print()
                    print("📚 TO TRAIN MODELS:")
                    print("   1. Collect clean training data for each application")
                    print("   2. Run training command for each app:")
                    for app in sorted(all_discovered_apps):
                        print(f"      python -m beam --train --app_name \"{app}\" -i /path/to/training/data")
                    print()
                    print("   3. Once trained, re-run BEAM for detection")
                    print()
                    print("📖 For detailed training instructions:")
                    print("   See models/custom_models/README.md")
                    print("=" * 60)
                    return
                else:
                    logger.info(f"Found matching models for: {sorted(matching_apps)}")
                    if missing_apps:
                        logger.warning(f"Missing models for: {sorted(missing_apps)}")

        for input_file in input_files:
            process_input_file(
                file_path=input_file,
                logger=logger,
                use_custom_models=use_custom_models,
            )
