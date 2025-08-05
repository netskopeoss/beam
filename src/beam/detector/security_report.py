"""Security Analysis Report Generator"""

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

import logging
from typing import Dict, List, Optional
from pathlib import Path
import json


class SecurityAnalysisReport:
    """
    Generates human-readable security analysis reports from BEAM feature data.
    """

    def __init__(self, prediction_dir: Path = None):
        self.logger = logging.getLogger(__name__)
        self.prediction_dir = prediction_dir

        # Thresholds for security analysis
        self.thresholds = {
            "automation_suspicion": 5.0,
            "suspicious_domain_ratio": 0.3,
            "bot_ratio": 0.1,
            "error_rate": 0.2,
            "mixed_content_risk": 0.1,
            "ua_entropy": 3.0,  # Lower entropy = more suspicious
            "external_domain_ratio": 0.8,
        }

    def analyze_security_features(
        self, summaries: List[Dict], prediction_dir: Optional[Path] = None
    ) -> Dict:
        """
        Analyze security features across all application summaries.

        Args:
            summaries (List[Dict]): List of application summary dictionaries
            prediction_dir (Optional[Path]): Directory containing ML prediction outputs

        Returns:
            Dict: Comprehensive security analysis
        """
        analysis = {
            "protocol_security": self._analyze_protocol_security(summaries),
            "header_fingerprinting": self._analyze_header_fingerprinting(summaries),
            "supply_chain_indicators": self._analyze_supply_chain_indicators(summaries),
            "behavioral_baselines": self._analyze_behavioral_baselines(summaries),
            "security_insights": self._generate_security_insights(summaries),
            "risk_assessment": self._assess_overall_risk(summaries),
        }

        # Add ML model explanations if available
        if prediction_dir:
            analysis["ml_explanations"] = self._load_ml_explanations(prediction_dir)

        return analysis

    def _analyze_protocol_security(self, summaries: List[Dict]) -> Dict:
        """Analyze TLS/HTTPS and protocol security features."""
        https_ratios = [s.get("https_ratio", 0) for s in summaries]
        http2_ratios = [s.get("http2_usage_ratio", 0) for s in summaries]
        mixed_content_risks = [s.get("mixed_content_risk", 0) for s in summaries]
        cert_depths = [s.get("cert_chain_depth_estimate", 0) for s in summaries]
        protocol_consistency = [s.get("protocol_consistency", 0) for s in summaries]

        return {
            "https_ratio": {
                "min": min(https_ratios) if https_ratios else 0,
                "max": max(https_ratios) if https_ratios else 0,
                "avg": sum(https_ratios) / len(https_ratios) if https_ratios else 0,
                "interpretation": self._interpret_https_ratio(
                    sum(https_ratios) / len(https_ratios) if https_ratios else 0
                ),
            },
            "http2_usage_ratio": {
                "avg": sum(http2_ratios) / len(http2_ratios) if http2_ratios else 0,
                "interpretation": self._interpret_http2_usage(
                    sum(http2_ratios) / len(http2_ratios) if http2_ratios else 0
                ),
            },
            "mixed_content_risk": {
                "max": max(mixed_content_risks) if mixed_content_risks else 0,
                "avg": sum(mixed_content_risks) / len(mixed_content_risks)
                if mixed_content_risks
                else 0,
                "interpretation": self._interpret_mixed_content(
                    max(mixed_content_risks) if mixed_content_risks else 0
                ),
            },
            "cert_chain_depth_estimate": {
                "min": min(cert_depths) if cert_depths else 0,
                "max": max(cert_depths) if cert_depths else 0,
                "interpretation": self._interpret_cert_depth(
                    min(cert_depths) if cert_depths else 0,
                    max(cert_depths) if cert_depths else 0,
                ),
            },
            "protocol_consistency": {
                "min": min(protocol_consistency) if protocol_consistency else 0,
                "interpretation": self._interpret_protocol_consistency(
                    min(protocol_consistency) if protocol_consistency else 0
                ),
            },
        }

    def _analyze_header_fingerprinting(self, summaries: List[Dict]) -> Dict:
        """Analyze HTTP header fingerprinting features."""
        ua_entropies = [s.get("ua_entropy", 0) for s in summaries]
        bot_ratios = [s.get("bot_ratio", 0) for s in summaries]
        ua_consistencies = [s.get("ua_consistency", 0) for s in summaries]
        referer_ratios = [s.get("referer_present_ratio", 0) for s in summaries]
        same_origin_ratios = [s.get("same_origin_referer_ratio", 0) for s in summaries]

        return {
            "ua_entropy": {
                "min": min(ua_entropies) if ua_entropies else 0,
                "max": max(ua_entropies) if ua_entropies else 0,
                "avg": sum(ua_entropies) / len(ua_entropies) if ua_entropies else 0,
                "interpretation": self._interpret_ua_entropy(
                    sum(ua_entropies) / len(ua_entropies) if ua_entropies else 0
                ),
            },
            "bot_ratio": {
                "max": max(bot_ratios) if bot_ratios else 0,
                "interpretation": self._interpret_bot_ratio(
                    max(bot_ratios) if bot_ratios else 0
                ),
            },
            "ua_consistency": {
                "min": min(ua_consistencies) if ua_consistencies else 0,
                "avg": sum(ua_consistencies) / len(ua_consistencies)
                if ua_consistencies
                else 0,
                "interpretation": self._interpret_ua_consistency(
                    sum(ua_consistencies) / len(ua_consistencies)
                    if ua_consistencies
                    else 0
                ),
            },
            "referer_patterns": {
                "referer_present_avg": sum(referer_ratios) / len(referer_ratios)
                if referer_ratios
                else 0,
                "same_origin_avg": sum(same_origin_ratios) / len(same_origin_ratios)
                if same_origin_ratios
                else 0,
                "interpretation": self._interpret_referer_patterns(
                    sum(referer_ratios) / len(referer_ratios) if referer_ratios else 0,
                    sum(same_origin_ratios) / len(same_origin_ratios)
                    if same_origin_ratios
                    else 0,
                ),
            },
        }

    def _analyze_supply_chain_indicators(self, summaries: List[Dict]) -> Dict:
        """Analyze supply chain specific security indicators."""
        external_ratios = [s.get("external_domain_ratio", 0) for s in summaries]
        thirdparty_ratios = [s.get("thirdparty_service_ratio", 0) for s in summaries]
        suspicious_ratios = [s.get("suspicious_domain_ratio", 0) for s in summaries]
        automation_suspicions = [s.get("automation_suspicion", 0) for s in summaries]
        api_ratios = [s.get("api_endpoint_ratio", 0) for s in summaries]
        dependency_complexities = [s.get("dependency_complexity", 0) for s in summaries]

        return {
            "dependency_tracking": {
                "external_domain_ratio_avg": sum(external_ratios) / len(external_ratios)
                if external_ratios
                else 0,
                "thirdparty_service_ratio_avg": sum(thirdparty_ratios)
                / len(thirdparty_ratios)
                if thirdparty_ratios
                else 0,
                "dependency_complexity_avg": sum(dependency_complexities)
                / len(dependency_complexities)
                if dependency_complexities
                else 0,
                "interpretation": self._interpret_dependency_tracking(
                    sum(external_ratios) / len(external_ratios)
                    if external_ratios
                    else 0,
                    sum(thirdparty_ratios) / len(thirdparty_ratios)
                    if thirdparty_ratios
                    else 0,
                ),
            },
            "suspicious_domains": {
                "suspicious_domain_ratio_max": max(suspicious_ratios)
                if suspicious_ratios
                else 0,
                "automation_suspicion_max": max(automation_suspicions)
                if automation_suspicions
                else 0,
                "api_endpoint_ratio_avg": sum(api_ratios) / len(api_ratios)
                if api_ratios
                else 0,
                "interpretation": self._interpret_suspicious_domains(
                    max(suspicious_ratios) if suspicious_ratios else 0,
                    max(automation_suspicions) if automation_suspicions else 0,
                ),
            },
        }

    def _analyze_behavioral_baselines(self, summaries: List[Dict]) -> Dict:
        """Analyze behavioral baseline features."""
        private_ip_ratios = [s.get("private_ip_ratio", 0) for s in summaries]
        error_rates = [s.get("error_rate", 0) for s in summaries]
        avg_bytes = [s.get("avg_bytes_per_request", 0) for s in summaries]

        return {
            "private_ip_ratio_avg": sum(private_ip_ratios) / len(private_ip_ratios)
            if private_ip_ratios
            else 0,
            "error_rate_max": max(error_rates) if error_rates else 0,
            "error_rate_avg": sum(error_rates) / len(error_rates) if error_rates else 0,
            "avg_bytes_per_request_range": {
                "min": min(avg_bytes) if avg_bytes else 0,
                "max": max(avg_bytes) if avg_bytes else 0,
            },
            "interpretation": self._interpret_behavioral_baselines(
                sum(error_rates) / len(error_rates) if error_rates else 0,
                max(error_rates) if error_rates else 0,
            ),
        }

    def _generate_security_insights(self, summaries: List[Dict]) -> List[Dict]:
        """Generate security insights from ModelExplainer explanations."""
        insights = []

        # Load ML explanations from prediction directory if available
        if (
            hasattr(self, "prediction_dir")
            and self.prediction_dir
            and self.prediction_dir.exists()
        ):
            try:
                for subdir in self.prediction_dir.iterdir():
                    if subdir.is_dir():
                        explanation_txt = subdir / "explanation.txt"
                        if explanation_txt.exists():
                            try:
                                with open(explanation_txt, "r") as f:
                                    explanation_text = f.read().strip()

                                # Extract domain from directory name (format: index_app_domain)
                                dir_parts = subdir.name.split("_")
                                if len(dir_parts) >= 3:
                                    app_name = dir_parts[1]
                                    domain = (
                                        "_".join(dir_parts[2:])
                                        .replace("_on_", " on ")
                                        .split(" on ")[0]
                                    )

                                    # Extract probability from explanation text
                                    probability = 0.0
                                    if "probability:" in explanation_text:
                                        try:
                                            prob_text = (
                                                explanation_text.split("probability:")[
                                                    1
                                                ]
                                                .split(")")[0]
                                                .strip()
                                            )
                                            probability = (
                                                float(prob_text.replace("%", ""))
                                                / 100.0
                                            )
                                        except:
                                            probability = 0.0

                                    # Determine severity based on probability
                                    if probability >= 0.95:
                                        severity = "HIGH"
                                    elif probability >= 0.8:
                                        severity = "MEDIUM"
                                    else:
                                        severity = "LOW"

                                    insights.append(
                                        {
                                            "type": "Supply Chain Compromise",
                                            "severity": severity,
                                            "application": app_name,
                                            "domain": domain,
                                            "key": f"{app_name} - {domain}",
                                            "details": explanation_text,
                                            "metric": "ml_prediction",
                                            "value": probability,
                                            "source": "ml_model",
                                        }
                                    )
                            except Exception:
                                # Continue if we can't parse this explanation
                                continue
            except Exception:
                # If we can't access the prediction directory, return empty insights
                pass

        return insights

    def _assess_overall_risk(self, summaries: List[Dict]) -> Dict:
        """Assess overall security risk level based on ML explanations."""
        risk_factors = []

        # Generate ML-based security insights first
        ml_insights = self._generate_security_insights(summaries)

        # Count applications with detected issues from ML analysis
        applications_with_issues = 0

        # Derive risk factors from ML insights
        for insight in ml_insights:
            if insight["severity"] == "HIGH":
                # Extract meaningful risk factors from ML analysis
                if "supply chain" in insight["type"].lower():
                    risk_factors.append(("Supply chain compromise detected", "HIGH"))
                elif "anomalous" in insight["type"].lower():
                    risk_factors.append(
                        ("Anomalous behavior patterns detected", "HIGH")
                    )
                else:
                    risk_factors.append((f"{insight['type']} detected", "HIGH"))
                applications_with_issues += 1
            elif insight["severity"] == "MEDIUM":
                risk_factors.append((f"{insight['type']} detected", "MEDIUM"))
                applications_with_issues += 1

        # If no ML insights available, fall back to basic feature analysis
        if not risk_factors:
            # Basic automation detection
            max_automation = max(
                [s.get("automation_suspicion", 0) for s in summaries], default=0
            )
            if max_automation > 10:
                risk_factors.append(("High automation patterns detected", "HIGH"))
            elif max_automation > 5:
                risk_factors.append(("Moderate automation patterns detected", "MEDIUM"))

            # Basic protocol security
            min_https = min([s.get("https_ratio", 1) for s in summaries], default=1)
            if min_https < 0.8:
                risk_factors.append(("Poor HTTPS adoption", "MEDIUM"))

        # Overall risk assessment - use highest prediction probability if available
        overall_risk = "MINIMAL"

        # Check if we have prediction probabilities to base the assessment on
        if (
            hasattr(self, "prediction_dir")
            and self.prediction_dir
            and self.prediction_dir.exists()
        ):
            max_probability = 0.0
            for subdir in self.prediction_dir.iterdir():
                if subdir.is_dir():
                    explanation_json = subdir / "explanation.json"
                    if explanation_json.exists():
                        try:
                            import json

                            with open(explanation_json, "r") as f:
                                data = json.load(f)
                                probability = float(data.get("probability", 0.0))
                                if probability > max_probability:
                                    max_probability = probability
                        except Exception:
                            pass

            # Set overall risk based on highest prediction probability
            if max_probability >= 0.95:
                overall_risk = "CRITICAL"
            elif max_probability >= 0.85:
                overall_risk = "HIGH"
            elif max_probability >= 0.70:
                overall_risk = "MEDIUM"
            elif max_probability >= 0.50:
                overall_risk = "LOW"
            else:
                overall_risk = "MINIMAL"
        else:
            # Fallback to risk factor counting if no predictions available
            high_risks = len([r for r in risk_factors if r[1] == "HIGH"])
            medium_risks = len([r for r in risk_factors if r[1] == "MEDIUM"])
            critical_risks = len([r for r in risk_factors if r[1] == "CRITICAL"])

            if critical_risks > 0:
                overall_risk = "CRITICAL"
            elif high_risks > 0:
                overall_risk = "HIGH"
            elif medium_risks > 1:
                overall_risk = "MEDIUM"
            elif medium_risks > 0:
                overall_risk = "LOW"
            else:
                overall_risk = "MINIMAL"

        return {
            "overall_risk_level": overall_risk,
            "risk_factors": risk_factors,
            "total_applications": len(summaries),
            "applications_with_issues": applications_with_issues,
        }

    # Interpretation methods
    def _interpret_https_ratio(self, ratio: float) -> str:
        if ratio >= 0.95:
            return "excellent HTTPS adoption"
        elif ratio >= 0.8:
            return "good HTTPS adoption"
        elif ratio >= 0.5:
            return "moderate HTTPS adoption"
        else:
            return "poor HTTPS adoption - security risk"

    def _interpret_http2_usage(self, ratio: float) -> str:
        if ratio == 0:
            return "no HTTP/2 usage detected"
        elif ratio < 0.3:
            return "limited HTTP/2 adoption"
        else:
            return "good HTTP/2 adoption"

    def _interpret_mixed_content(self, risk: float) -> str:
        if risk == 0:
            return "no mixed content issues"
        elif risk < 0.1:
            return "minimal mixed content risk"
        else:
            return "mixed content security risk detected"

    def _interpret_cert_depth(self, min_depth: float, max_depth: float) -> str:
        if max_depth <= 2:
            return "simple certificate structure"
        elif max_depth <= 4:
            return "moderate certificate complexity"
        else:
            return "complex certificate chain structure"

    def _interpret_protocol_consistency(self, consistency: float) -> str:
        if consistency >= 0.95:
            return "very consistent protocol usage"
        elif consistency >= 0.8:
            return "mostly consistent protocol usage"
        else:
            return "inconsistent protocol usage - potential security concern"

    def _interpret_ua_entropy(self, entropy: float) -> str:
        if entropy > 5:
            return "high User-Agent randomness (normal)"
        elif entropy > 3:
            return "moderate User-Agent variety"
        else:
            return "low User-Agent entropy - potentially suspicious"

    def _interpret_bot_ratio(self, ratio: float) -> str:
        if ratio == 0:
            return "no bot signatures detected"
        elif ratio < 0.1:
            return "minimal bot activity"
        else:
            return "significant bot activity detected"

    def _interpret_ua_consistency(self, consistency: float) -> str:
        if consistency > 0.95:
            return "very consistent User-Agent strings"
        elif consistency > 0.8:
            return "mostly consistent User-Agent usage"
        else:
            return "inconsistent User-Agent patterns"

    def _interpret_referer_patterns(
        self, referer_ratio: float, same_origin_ratio: float
    ) -> str:
        if same_origin_ratio > 0.8:
            return "mostly same-origin requests (normal)"
        elif same_origin_ratio > 0.5:
            return "mixed cross-origin patterns"
        else:
            return "high cross-origin activity"

    def _interpret_dependency_tracking(
        self, external_ratio: float, thirdparty_ratio: float
    ) -> str:
        if external_ratio > 0.8:
            desc = "high external dependency usage"
        else:
            desc = "moderate external dependencies"

        if thirdparty_ratio > 0.3:
            desc += ", significant third-party services"
        elif thirdparty_ratio > 0:
            desc += ", some third-party services"
        else:
            desc += ", minimal third-party services"

        return desc

    def _interpret_suspicious_domains(
        self, suspicious_ratio: float, automation_suspicion: float
    ) -> str:
        issues = []
        if suspicious_ratio > 0.5:
            issues.append("high suspicious domain activity")
        elif suspicious_ratio > 0.1:
            issues.append("some suspicious domain patterns")

        if automation_suspicion > 10:
            issues.append("very high automation suspicion")
        elif automation_suspicion > 5:
            issues.append("moderate automation suspicion")

        if not issues:
            return "no significant suspicious activity detected"
        else:
            return ", ".join(issues)

    def _interpret_behavioral_baselines(
        self, avg_error_rate: float, max_error_rate: float
    ) -> str:
        if max_error_rate > 0.3:
            return "high error rates detected - potential issues"
        elif avg_error_rate > 0.1:
            return "moderate error activity"
        else:
            return "low error rates - normal behavior"

    def _load_ml_explanations(self, prediction_dir: Path) -> List[Dict]:
        """Load ML model explanations from prediction directory."""
        explanations = []

        if not prediction_dir or not isinstance(prediction_dir, Path):
            return explanations

        try:
            # Iterate through prediction subdirectories
            for subdir in prediction_dir.iterdir():
                if subdir.is_dir():
                    # Try to load explanation.json
                    explanation_json = subdir / "explanation.json"
                    explanation_txt = subdir / "explanation.txt"

                    if explanation_json.exists():
                        try:
                            with open(explanation_json, "r") as f:
                                exp_data = json.load(f)
                                explanations.append(exp_data)
                        except Exception as e:
                            self.logger.debug(f"Failed to load {explanation_json}: {e}")
                    elif explanation_txt.exists():
                        # Fallback to text explanation
                        try:
                            with open(explanation_txt, "r") as f:
                                text = f.read()
                                explanations.append(
                                    {
                                        "domain": subdir.name.split("_", 1)[-1]
                                        if "_" in subdir.name
                                        else "unknown",
                                        "explanation": text,
                                        "source": "text",
                                    }
                                )
                        except Exception as e:
                            self.logger.debug(f"Failed to load {explanation_txt}: {e}")
        except Exception as e:
            self.logger.debug(
                f"Failed to load ML explanations from {prediction_dir}: {e}"
            )

        return explanations

    def format_security_report(self, analysis: Dict) -> str:
        """Format the security analysis into a human-readable report."""
        report = []

        report.append("🔒 BEAM Security Analysis Report")
        report.append("=" * 50)
        report.append("")

        # Protocol Security Analysis
        report.append("🌐 TLS/HTTPS Analysis:")
        protocol = analysis["protocol_security"]
        report.append(
            f"  - HTTPS ratio: {protocol['https_ratio']['avg']:.2f} ({protocol['https_ratio']['interpretation']})"
        )
        report.append(
            f"  - HTTP/2 usage: {protocol['http2_usage_ratio']['avg']:.2f} ({protocol['http2_usage_ratio']['interpretation']})"
        )
        report.append(
            f"  - Mixed content risk: {protocol['mixed_content_risk']['max']:.2f} ({protocol['mixed_content_risk']['interpretation']})"
        )
        report.append(
            f"  - Certificate chain depth: {protocol['cert_chain_depth_estimate']['min']:.0f}-{protocol['cert_chain_depth_estimate']['max']:.0f} ({protocol['cert_chain_depth_estimate']['interpretation']})"
        )
        report.append(
            f"  - Protocol consistency: {protocol['protocol_consistency']['min']:.2f} ({protocol['protocol_consistency']['interpretation']})"
        )
        report.append("")

        # Header Fingerprinting
        report.append("🔍 HTTP Header Fingerprinting:")
        headers = analysis["header_fingerprinting"]
        report.append(
            f"  - User-Agent entropy: {headers['ua_entropy']['avg']:.2f} ({headers['ua_entropy']['interpretation']})"
        )
        report.append(
            f"  - Bot activity ratio: {headers['bot_ratio']['max']:.2f} ({headers['bot_ratio']['interpretation']})"
        )
        report.append(
            f"  - User-Agent consistency: {headers['ua_consistency']['avg']:.2f} ({headers['ua_consistency']['interpretation']})"
        )
        report.append(
            f"  - Referrer patterns: {headers['referer_patterns']['interpretation']}"
        )
        report.append("")

        # Supply Chain Indicators
        report.append("🔗 Supply Chain Security Indicators:")
        supply_chain = analysis["supply_chain_indicators"]

        report.append("  Dependency Tracking:")
        dep = supply_chain["dependency_tracking"]
        report.append(
            f"    - External domain ratio: {dep['external_domain_ratio_avg']:.2f}"
        )
        report.append(
            f"    - Third-party service ratio: {dep['thirdparty_service_ratio_avg']:.2f}"
        )
        report.append(
            f"    - Dependency complexity: {dep['dependency_complexity_avg']:.1f}"
        )
        report.append(f"    - Assessment: {dep['interpretation']}")

        report.append("  Suspicious Domain Detection:")
        susp = supply_chain["suspicious_domains"]
        report.append(
            f"    - Suspicious domain ratio: {susp['suspicious_domain_ratio_max']:.2f}"
        )
        report.append(
            f"    - Automation suspicion: {susp['automation_suspicion_max']:.1f}"
        )
        report.append(f"    - API endpoint ratio: {susp['api_endpoint_ratio_avg']:.2f}")
        report.append(f"    - Assessment: {susp['interpretation']}")
        report.append("")

        # Behavioral Baselines
        report.append("📊 Behavioral Baselines:")
        behavioral = analysis["behavioral_baselines"]
        report.append(f"  - Private IP ratio: {behavioral['private_ip_ratio_avg']:.2f}")
        report.append(
            f"  - Error rate: {behavioral['error_rate_avg']:.2f} (max: {behavioral['error_rate_max']:.2f})"
        )
        report.append(
            f"  - Data volume range: {behavioral['avg_bytes_per_request_range']['min']:.0f}-{behavioral['avg_bytes_per_request_range']['max']:.0f} bytes/request"
        )
        report.append(f"  - Assessment: {behavioral['interpretation']}")
        report.append("")

        # Security Insights
        insights = analysis["security_insights"]
        if insights:
            report.append("🚨 Security Insights Detected:")
            for i, insight in enumerate(insights, 1):
                report.append(
                    f"  {i}. {insight['type']} [{insight['severity']}]: {insight['details']}"
                )
                if insight.get("domain"):
                    report.append(
                        f"     Domain: {insight['domain']} | Application: {insight['application']}"
                    )
            report.append("")

        # Risk Assessment
        report.append("⚠️  Overall Risk Assessment:")
        risk = analysis["risk_assessment"]
        report.append(f"  - Risk Level: {risk['overall_risk_level']}")
        report.append(f"  - Applications analyzed: {risk['total_applications']}")
        report.append(
            f"  - Applications with issues: {risk['applications_with_issues']}"
        )

        if risk["risk_factors"]:
            report.append("  - Risk factors identified:")
            for factor, severity in risk["risk_factors"]:
                report.append(f"    • {factor} [{severity}]")

        # ML Model Explanations
        if "ml_explanations" in analysis and analysis["ml_explanations"]:
            report.append("")
            report.append("🤖 Machine Learning Model Analysis:")
            report.append("-" * 50)

            for i, exp in enumerate(analysis["ml_explanations"], 1):
                if exp.get("is_anomaly", False):
                    report.append(f"  Anomaly {i}:")
                    report.append(f"    Domain: {exp.get('domain', 'unknown')}")
                    report.append(
                        f"    Application: {exp.get('application', 'unknown')}"
                    )
                    report.append(
                        f"    Confidence: {float(exp.get('probability', 0)):.1%}"
                    )

                    # Include the explanation text
                    if "explanation" in exp:
                        report.append("    Explanation:")
                        # Split and indent the explanation
                        explanation_lines = exp["explanation"].strip().split("\n")
                        for line in explanation_lines[:5]:  # First 5 lines
                            if line.strip():
                                report.append(f"      {line.strip()}")

                    # Include top features if available
                    if "top_features" in exp and exp["top_features"]:
                        report.append("    Top Contributing Features:")
                        for j, feature in enumerate(exp["top_features"][:3], 1):
                            report.append(
                                f"      {j}. {feature['feature']}: {feature['feature_value']:.2f} "
                                f"(SHAP: {feature['shap_value']:+.3f})"
                            )

                    report.append("")

        return "\n".join(report)


def generate_security_report(
    summaries: List[Dict], prediction_dir: Optional[Path] = None
) -> str:
    """
    Generate a comprehensive security analysis report from BEAM summaries.

    Args:
        summaries (List[Dict]): List of application summary dictionaries
        prediction_dir (Optional[Path]): Directory containing ML prediction outputs

    Returns:
        str: Formatted security analysis report
    """
    analyzer = SecurityAnalysisReport(prediction_dir)
    analysis = analyzer.analyze_security_features(summaries, prediction_dir)
    return analyzer.format_security_report(analysis)
