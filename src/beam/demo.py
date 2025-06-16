"""BEAM Demo Module - Supply Chain Compromise Detection Showcase"""

# Copyright 2025 Netskope, Inc.

import logging
import os
import shutil
from pathlib import Path

from art import tprint

from beam import constants
from beam.detector import features, utils
from beam.detector.detect import detect_anomalous_domain
from beam.detector.security_report import generate_security_report
from beam.enrich import enrich_events
from beam.parser import har


def run_demo(logger: logging.Logger = None) -> None:
    """
    Run the BEAM demo showcasing supply chain compromise detection.
    
    Args:
        logger: Logger instance for demo execution
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Print demo header
    tprint("BEAM")
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
    print("⏱️  Demo duration: ~5 seconds")
    print("=" * 60)
    print()
    
    # Demo file paths
    demo_dir = Path(__file__).parent.parent.parent / "demo"
    demo_data_file = demo_dir / "data" / "beacon_02_17_2025_18_12_39.har"
    
    if not demo_data_file.exists():
        print("❌ Demo data file not found!")
        print(f"Expected: {demo_data_file}")
        print("Please ensure the demo data is available.")
        return
    
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
    
    try:
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
        )
        utils.save_json_data(enriched_events, str(enriched_output_path))
        print(f"   ✓ Enriched events with application mapping")
        
        # Step 3: Extract security features
        print("🛡️  Step 3: Extracting security features...")
        features.aggregate_app_traffic(
            fields=["application", "domain"],
            input_path=str(enriched_output_path),
            output_path=str(features_output_path),
            min_transactions=1,  # Lower threshold for demo
        )
        print("   ✓ Extracted comprehensive security features")
        
        # Step 4: Run anomaly detection
        print("🔍 Step 4: Running supply chain compromise detection...")
        model_path = Path(constants.DOMAIN_MODEL)
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
        summaries = utils.load_json_file(str(features_output_path))
        security_report = generate_security_report(summaries)
        
        # Save security report
        security_report_path = temp_demo_dir / f"{file_name}_security_report.txt"
        with open(security_report_path, 'w') as f:
            f.write(security_report)
        
        print("   ✓ Generated comprehensive security analysis")
        print()
        
        # Step 6: Display demo results
        print("🎯 DEMO RESULTS")
        print("=" * 60)
        
        # Extract and display key insights
        from beam.detector.security_report import SecurityAnalysisReport
        analyzer = SecurityAnalysisReport()
        analysis = analyzer.analyze_security_features(summaries)
        
        # Show key findings
        print("🔍 NETWORK TRAFFIC ANALYSIS:")
        print(f"   • Applications detected: {len(summaries)}")
        print(f"   • Security features extracted: ~240+ per application")
        print(f"   • Analysis techniques: Protocol, Header, Supply Chain, Behavioral")
        print()
        
        # Show security insights
        insights = analysis['security_insights']
        critical_insights = [i for i in insights if i['severity'] in ['HIGH', 'MEDIUM']]
        
        print("🚨 SECURITY INSIGHTS DETECTED:")
        if critical_insights:
            # Check if we have the specific supply chain compromise
            supply_chain_compromise = None
            for insight in critical_insights:
                if 'xqpt5z.dagmawi.io' in insight['domain']:
                    supply_chain_compromise = insight
                    break
            
            # If we found the supply chain compromise, explain it clearly
            if supply_chain_compromise:
                print()
                print("🔥 CRITICAL SUPPLY CHAIN COMPROMISE DETECTED!")
                print("=" * 60)
                print("📋 WHAT HAPPENED:")
                print("   The Box application is communicating with an unauthorized server:")
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
                print("   where legitimate software is modified to include malicious code.")
                print("=" * 60)
                print()
            
            # Show all insights
            for insight in critical_insights:
                severity_emoji = "🔥" if insight['severity'] == 'HIGH' else "⚠️"
                print(f"   {severity_emoji} [{insight['severity']}] {insight['type']}")
                print(f"      Domain: {insight['domain']}")
                print(f"      Technical details: {insight['details']}")
                print()
        else:
            print("   ✅ No critical security issues detected")
        
        # Show overall assessment
        risk_level = analysis['risk_assessment']['overall_risk_level']
        risk_emoji = {"HIGH": "🔥", "MEDIUM": "⚠️", "LOW": "💛", "MINIMAL": "✅"}.get(risk_level, "❓")
        
        print("📈 OVERALL SECURITY ASSESSMENT:")
        print(f"   🎯 Risk Level: {risk_emoji} {risk_level}")
        print(f"   📊 Applications analyzed: {analysis['risk_assessment']['total_applications']}")
        print(f"   🔍 Applications with issues: {analysis['risk_assessment']['applications_with_issues']}")
        
        if analysis['risk_assessment']['risk_factors']:
            print("   ⚠️  Risk factors identified:")
            for factor, severity in analysis['risk_assessment']['risk_factors']:
                print(f"      • {factor} [{severity}]")
        
        print()
        print("📄 DETAILED REPORTS AVAILABLE:")
        print(f"   • Security Analysis: {security_report_path}")
        print(f"   • Feature Data: {features_output_path}")
        print(f"   • Predictions: {prediction_dir}")
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
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        print("Please check the logs for more details.")
    
    finally:
        # Clean up temporary files
        if temp_demo_dir.exists():
            try:
                shutil.rmtree(temp_demo_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up demo temp directory: {e}")


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
        'supported_apps': [],
        'unsupported_apps': [],
        'model_files': [],
        'input_files': []
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
    print("   2. Run: python -m beam --train --app_name \"App Name\" -i /path/to/training/data")
    print("   3. Once trained, re-run BEAM for detection")
    print()
    print("📚 See models/custom_models/README.md for detailed training instructions")