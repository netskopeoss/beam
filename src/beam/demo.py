"""BEAM Demo Module - Supply Chain Compromise Detection Showcase"""

# Copyright 2025 Netskope, Inc.

import logging
import shutil
from pathlib import Path
from typing import Optional

# Make MultiHotEncoder available in __main__ context for pickle loading
import __main__
from beam import constants
from beam.detector import features, utils
from beam.detector.detect import MultiHotEncoder, detect_anomalous_domain
from beam.detector.security_report import generate_security_report
from beam.enrich import enrich_events
from beam.parser import har

__main__.MultiHotEncoder = MultiHotEncoder


def print_demo_header() -> None:
    """Print the demo introduction and header."""
    print("🔒 BEAM Demo - Supply Chain Compromise Detection")
    print("=" * 60)
    print("📖 WHAT IS A SUPPLY CHAIN COMPROMISE?")
    print("   When hackers inject malicious code into legitimate software,")
    print("   turning trusted applications into data-stealing trojans.")
    print()
    print("🎯 WHAT THIS DEMO WILL SHOW:")
    print("   • Real network traffic from the Box cloud storage application")
    print("   • Hidden malware communicating with hacker-controlled servers")
    print("   • How BEAM's AI detects this suspicious behavior automatically")
    print()
    print("⏱️  Demo duration: ~20-30 seconds")
    print("=" * 60)
    print()


def setup_demo_environment() -> tuple[Path, Path, Path, Path, Path, Path]:
    """
    Set up the demo environment and file paths.

    Returns:
        Tuple of paths: (demo_data_file, temp_demo_dir, demo_input_path,
                        parsed_output_path, enriched_output_path, features_output_path)
    """
    # Demo file paths
    demo_dir = Path(__file__).parent.parent.parent / "demo"
    demo_data_file = demo_dir / "data" / "beacon_02_17_2025_18_12_39.har"

    if not demo_data_file.exists():
        print("❌ Demo data file not found!")
        print(f"Expected: {demo_data_file}")
        print("Please ensure the demo data is available.")
        raise FileNotFoundError(f"Demo data file not found: {demo_data_file}")

    print(f"📁 Processing demo data: {demo_data_file.name}")
    print()

    # Create temporary working directories for demo
    temp_demo_dir = constants.DATA_DIR / "demo_temp"
    temp_demo_dir.mkdir(exist_ok=True)

    # Set up demo paths
    file_name = "beacon_demo"
    demo_input_path = temp_demo_dir / f"{file_name}.har"
    parsed_output_path = temp_demo_dir / f"{file_name}_parsed.json"
    enriched_output_path = temp_demo_dir / f"{file_name}_enriched.json"
    features_output_path = temp_demo_dir / f"{file_name}_features.json"

    return (
        demo_data_file,
        temp_demo_dir,
        demo_input_path,
        parsed_output_path,
        enriched_output_path,
        features_output_path,
    )


def process_demo_data(
    demo_data_file: Path,
    demo_input_path: Path,
    parsed_output_path: Path,
    enriched_output_path: Path,
    features_output_path: Path,
) -> None:
    """
    Process the demo data through the BEAM pipeline.

    Args:
        demo_data_file: Path to the original demo data file
        demo_input_path: Path to the temporary demo input file
        parsed_output_path: Path to save parsed data
        enriched_output_path: Path to save enriched data
        features_output_path: Path to save extracted features
    """
    # Copy demo file to temp location
    shutil.copy2(demo_data_file, demo_input_path)

    # Step 1: Parse HAR file
    print("🔍 Step 1: Parsing network traffic data...")
    parsed_transactions = har.parse_har_log(str(demo_input_path))

    # Save parsed data
    parsed_data = [t.model_dump() for t in parsed_transactions]
    utils.save_json_data(parsed_data, str(parsed_output_path))
    print(f"   ✓ Parsed {len(parsed_transactions)} network transactions")

    # Step 2: Enrich events with application mapping
    print("🔗 Step 2: Enriching events with application intelligence...")
    enriched_events = enrich_events(
        input_path=str(parsed_output_path),
        db_path=str(constants.DB_PATH),
        cloud_domains_file_path=str(constants.CLOUD_DOMAINS_FILE),
        key_domains_file_path=str(constants.KEY_DOMAINS_FILE),
        llm_api_key=constants.GEMINI_API_KEY,
        use_local_llm=constants.USE_LOCAL_LLM,
    )
    utils.save_json_data(enriched_events, str(enriched_output_path))
    print("   ✓ Enriched events with application mapping")

    # Step 3: Extract security features
    print("🛡️  Step 3: Extracting security features...")
    features.aggregate_app_traffic(
        fields=["application", "domain"],
        input_path=str(enriched_output_path),
        output_path=str(features_output_path),
        min_transactions=10,  # Minimum needed for statistical features
    )
    print("   ✓ Extracted comprehensive security features")


def run_detection_analysis(features_output_path: Path, temp_demo_dir: Path) -> Path:
    """
    Run anomaly detection and generate security report.

    Args:
        features_output_path: Path to the extracted features
        temp_demo_dir: Temporary directory for demo files

    Returns:
        Path to the security report file
    """
    # Step 4: Run anomaly detection
    print("🔍 Step 4: Running supply chain compromise detection...")
    from path import Path as PathLib

    model_path = PathLib(constants.DOMAIN_MODEL)
    prediction_dir = temp_demo_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    detect_anomalous_domain(
        input_path=str(features_output_path),
        domain_model_path=model_path,
        app_prediction_dir=str(prediction_dir),
    )
    print("   ✓ Completed anomaly detection analysis")

    # Step 5: Generate security analysis report
    print("📊 Step 5: Generating security analysis report...")
    summaries_data = utils.load_json_file(str(features_output_path))

    # Ensure summaries is a list
    if isinstance(summaries_data, dict):
        summaries = [summaries_data]
    else:
        summaries = summaries_data

    security_report = generate_security_report(summaries, prediction_dir)

    # Save security report
    file_name = "beacon_demo"
    security_report_path = temp_demo_dir / f"{file_name}_security_report.txt"
    with open(security_report_path, "w") as f:
        f.write(security_report)

    print("   ✓ Generated comprehensive security analysis")
    print()

    return security_report_path


def display_demo_results(
    features_output_path: Path, security_report_path: Path, prediction_dir: Path
) -> None:
    """
    Display the demo results and analysis.

    Args:
        features_output_path: Path to the extracted features
        security_report_path: Path to the security report
        prediction_dir: Directory containing predictions
    """
    print("🎯 DEMO RESULTS")
    print("=" * 60)

    # Extract and display key insights
    from beam.detector.security_report import SecurityAnalysisReport

    summaries_data = utils.load_json_file(str(features_output_path))

    # Ensure summaries is a list
    if isinstance(summaries_data, dict):
        summaries = [summaries_data]
    else:
        summaries = summaries_data

    analyzer = SecurityAnalysisReport()
    analysis = analyzer.analyze_security_features(summaries, prediction_dir)

    # Show key findings
    print("🔍 NETWORK TRAFFIC ANALYSIS:")
    print(f"   • Applications detected: {len(summaries)}")
    print("   • Security features extracted: ~240+ per application")
    print("   • Analysis techniques: Protocol, Header, Supply Chain, Behavioral")
    print()

    # Show security insights
    insights = analysis["security_insights"]
    critical_insights = [i for i in insights if i["severity"] in ["HIGH", "MEDIUM"]]

    display_security_insights(critical_insights, prediction_dir)
    display_overall_assessment(analysis)
    display_available_reports(
        security_report_path, features_output_path, prediction_dir
    )


def display_security_insights(critical_insights: list, prediction_dir: Path = None) -> None:
    """
    Display security insights and supply chain compromise details.

    Args:
        critical_insights: List of critical security insights
        prediction_dir: Directory containing prediction outputs with explanations
    """
    print("🚨 SECURITY INSIGHTS DETECTED:")
    if critical_insights:
        # Check if we have the specific supply chain compromise
        supply_chain_compromise = None
        for insight in critical_insights:
            if "xqpt5z.dagmawi.io" in insight["domain"]:
                supply_chain_compromise = insight
                break

        # If we found the supply chain compromise, explain it clearly
        if supply_chain_compromise:
            print()
            print("🔥 CRITICAL SUPPLY CHAIN COMPROMISE DETECTED!")
            print("=" * 60)
            print("📋 WHAT HAPPENED:")
            print(
                "   The Box application is communicating with an unauthorized server:"
            )
            print(f"   • Suspicious domain: {supply_chain_compromise['domain']}")
            print("   • Expected behavior: Box should only talk to *.box.com servers")
            print("   • Actual behavior: Box is sending data to an unknown domain")
            print()
            print("🔍 WHAT OUR ANALYSIS FOUND:")
            print("   • Regular, automated communication pattern (every 5 seconds)")
            print("   • Not normal user behavior - this is programmatic")
            print("   • High automation score (22.2) indicates malicious bot activity")
            print()
            print("⚠️  WHAT THIS MEANS:")
            print("   This is likely malicious code injected into the Box application")
            print("   that is stealing data or establishing a backdoor connection.")
            print("   This type of attack is called a 'supply chain compromise'")
            
            # Try to load and display SHAP-based explanation if available
            if prediction_dir and prediction_dir.exists():
                try:
                    # Find the prediction subdirectory for this domain
                    for subdir in prediction_dir.iterdir():
                        if subdir.is_dir():
                            explanation_file = subdir / "explanation.txt"
                            if explanation_file.exists():
                                with open(explanation_file, 'r') as f:
                                    shap_explanation = f.read()
                                
                                print()
                                print("🤖 AI MODEL EXPLANATION:")
                                print("-" * 60)
                                # Display the first part of the explanation
                                lines = shap_explanation.strip().split('\n')
                                for line in lines[:10]:  # Show first 10 lines
                                    if line.strip():
                                        print(f"   {line}")
                                print("-" * 60)
                                break
                except Exception as e:
                    # Silently continue if we can't read the explanation
                    pass
            print("   where legitimate software is modified to include malicious code.")
            print("=" * 60)
            print()

        # Show all insights
        for insight in critical_insights:
            severity_emoji = "🔥" if insight["severity"] == "HIGH" else "⚠️"
            print(f"   {severity_emoji} [{insight['severity']}] {insight['type']}")
            print(f"      Domain: {insight['domain']}")
            print(f"      Technical details: {insight['details']}")
            print()
    else:
        print("   ✅ No critical security issues detected")


def display_overall_assessment(analysis: dict) -> None:
    """
    Display the overall security assessment.

    Args:
        analysis: Security analysis results
    """
    # Show overall assessment
    risk_level = analysis["risk_assessment"]["overall_risk_level"]
    risk_emoji = {"HIGH": "🔥", "MEDIUM": "⚠️", "LOW": "💛", "MINIMAL": "✅"}.get(
        risk_level, "❓"
    )

    print("📈 OVERALL SECURITY ASSESSMENT:")
    print(f"   🎯 Risk Level: {risk_emoji} {risk_level}")
    print(
        f"   📊 Applications analyzed: {analysis['risk_assessment']['total_applications']}"
    )
    print(
        f"   🔍 Applications with issues: {analysis['risk_assessment']['applications_with_issues']}"
    )

    if analysis["risk_assessment"]["risk_factors"]:
        print("   ⚠️  Risk factors identified:")
        for factor, severity in analysis["risk_assessment"]["risk_factors"]:
            print(f"      • {factor} [{severity}]")

    print()


def display_available_reports(
    security_report_path: Path, features_output_path: Path, prediction_dir: Path
) -> None:
    """
    Display information about available reports and demo conclusion.

    Args:
        security_report_path: Path to the security report
        features_output_path: Path to the features data
        prediction_dir: Directory containing predictions
    """
    # Convert container paths to host paths for user-friendly display
    def convert_container_path_to_host_path(container_path: str) -> str:
        """Convert container path to host filesystem path for user-friendly console output."""
        if container_path.startswith("/app/"):
            return "./demo_results/" + container_path.split("/")[-1]  # Put demo files in demo_results/
        return container_path
    
    print("📄 DETAILED REPORTS AVAILABLE:")
    print(f"   • Security Analysis: {convert_container_path_to_host_path(str(security_report_path))}")
    print(f"   • Feature Data: {convert_container_path_to_host_path(str(features_output_path))}")
    print(f"   • Predictions: {convert_container_path_to_host_path(str(prediction_dir))}")
    print()

    # Demo conclusion
    print("🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("📚 WHAT YOU'VE LEARNED:")
    print()
    print("   BEAM uses machine learning to detect when trusted applications")
    print("   start behaving suspiciously. In this demo:")
    print()
    print("   1. We analyzed network traffic from the Box application")
    print("   2. BEAM's ML models know how Box normally behaves")
    print("   3. We detected unusual communication to 'xqpt5z.dagmawi.io'")
    print("   4. The pattern (every 5 seconds) revealed automated malware")
    print()
    print("🛡️  WHY THIS MATTERS:")
    print("   Supply chain attacks are increasingly common and hard to detect.")
    print("   Attackers compromise legitimate software to reach many victims.")
    print("   BEAM helps you catch these compromises before damage occurs.")
    print()
    print("🚀 READY TO PROTECT YOUR APPLICATIONS?")
    print("   1. Place your .har or .pcap files in data/input/")
    print("   2. Train models for your specific applications")
    print("   3. Run: python -m beam --use_custom_models")
    print()
    print("💡 TIP: Start by training models on clean traffic, then use them")
    print("   to detect anomalies in production traffic.")


def cleanup_demo_files(temp_demo_dir: Path, logger: logging.Logger) -> None:
    """
    Clean up temporary demo files.

    Args:
        temp_demo_dir: Temporary directory to clean up
        logger: Logger instance for error reporting
    """
    if temp_demo_dir.exists():
        try:
            shutil.rmtree(temp_demo_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up demo temp directory: {e}")


def setup_demo_logging():
    """Setup logging for demo with custom log path if specified."""
    import os
    from pathlib import Path
    
    custom_log_path = os.environ.get('BEAM_LOG_PATH')
    
    if custom_log_path:
        # Create the directory if it doesn't exist
        log_dir = Path(custom_log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging programmatically (file only)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(custom_log_path)
            ],
            force=True  # Override any existing configuration
        )
    else:
        # Use the default logging configuration
        import logging.config as log_config
        from beam.constants import LOG_CONFIG
        log_config.fileConfig(LOG_CONFIG)


def run_demo(
    logger: Optional[logging.Logger] = None, preserve_results: Optional[bool] = None
) -> None:
    """
    Run the BEAM demo showcasing supply chain compromise detection.

    Args:
        logger: Logger instance for demo execution
        preserve_results: Whether to preserve demo results (auto-detects Docker if None)
    """
    # Setup logging if not already configured
    setup_demo_logging()
    
    if logger is None:
        logger = logging.getLogger(__name__)

    # Auto-detect if we're running in Docker and should preserve results
    if preserve_results is None:
        import os

        preserve_results = (
            os.path.exists("/.dockerenv")
            or os.environ.get("DOCKER_DEMO", "false").lower() == "true"
        )

    print_demo_header()

    temp_demo_dir = None

    try:
        # Set up demo environment
        (
            demo_data_file,
            temp_demo_dir,
            demo_input_path,
            parsed_output_path,
            enriched_output_path,
            features_output_path,
        ) = setup_demo_environment()

        # Process the demo data through BEAM pipeline
        process_demo_data(
            demo_data_file,
            demo_input_path,
            parsed_output_path,
            enriched_output_path,
            features_output_path,
        )

        # Run detection and generate reports
        security_report_path = run_detection_analysis(
            features_output_path, temp_demo_dir
        )

        # Display results
        prediction_dir = temp_demo_dir / "predictions"
        display_demo_results(features_output_path, security_report_path, prediction_dir)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        print("Please check the logs for more details.")

    finally:
        # Clean up temporary files unless preserving results
        if temp_demo_dir is not None and not preserve_results:
            cleanup_demo_files(temp_demo_dir, logger)
        elif temp_demo_dir is not None and preserve_results:
            logger.info(f"Demo results preserved in: {temp_demo_dir}")


def check_for_supported_applications(input_dir: Path, model_dir: Path) -> dict:
    """
    Check which applications in input data have corresponding trained models.

    Args:
        input_dir: Directory containing input files
        model_dir: Directory containing trained models

    Returns:
        dict: Analysis of supported vs unsupported applications
    """
    # This would be implemented to scan input files and check against available models
    # For now, return a placeholder structure
    return {
        "supported_apps": [],
        "unsupported_apps": [],
        "model_files": [],
        "input_files": [],
    }


def suggest_training_workflow(unsupported_apps: list) -> None:
    """
    Provide guidance on training models for unsupported applications.

    Args:
        unsupported_apps: List of applications without trained models
    """
    if not unsupported_apps:
        return

    print("🎓 TRAINING REQUIRED")
    print("=" * 60)
    print("The following applications were detected but don't have trained models:")

    for app in unsupported_apps:
        print(f"   • {app}")

    print()
    print("To analyze these applications, you need to train models first:")
    print("   1. Collect clean training data for each application")
    print(
        '   2. Run: python -m beam --train --app_name "App Name" -i /path/to/training/data'
    )
    print("   3. Once trained, re-run BEAM for detection")
    print()
    print("📚 See models/custom_models/README.md for detailed training instructions")
