"""Model Explainer Module - Generates human-readable explanations for predictions"""

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

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from beam import constants


# Feature groups for better explanation organization
FEATURE_GROUPS = {
    "traffic_volume": {
        "features": ["sum_client_bytes", "sum_server_bytes", "avg_client_bytes", 
                    "avg_server_bytes", "total_data_volume", "avg_bytes_per_request",
                    "min_client_bytes", "max_client_bytes", "range_client_bytes",
                    "min_server_bytes", "max_server_bytes", "range_server_bytes",
                    "p25_client_bytes", "p75_client_bytes", "p90_client_bytes",
                    "p95_client_bytes", "p99_client_bytes", "iqr_client_bytes",
                    "p25_server_bytes", "p75_server_bytes", "p90_server_bytes",
                    "p95_server_bytes", "p99_server_bytes", "iqr_server_bytes",
                    "cv_client_bytes", "cv_server_bytes", "skewness_client_bytes",
                    "skewness_server_bytes", "kurtosis_client_bytes", "kurtosis_server_bytes",
                    "outlier_ratio_client_bytes", "outlier_ratio_server_bytes",
                    "mad_client_bytes", "mad_server_bytes", "robust_cv_client_bytes",
                    "robust_cv_server_bytes"],
        "description": "data transfer patterns"
    },
    "time_patterns": {
        "features": ["avg_time_interval_sec", "std_time_interval_sec", "median_time_interval_sec",
                    "burst_ratio", "interval_entropy", "interval_regularity", "night_activity_ratio",
                    "hour_entropy", "avg_interval_sec", "peak_hour_concentration",
                    "min_time_interval_sec", "max_time_interval_sec", "range_time_interval_sec",
                    "p25_time_interval_sec", "p75_time_interval_sec", "p90_time_interval_sec",
                    "p95_time_interval_sec", "p99_time_interval_sec", "iqr_time_interval_sec",
                    "cv_time_interval_sec", "skewness_time_interval_sec", "kurtosis_time_interval_sec",
                    "outlier_ratio_time_interval_sec", "mad_time_interval_sec", "robust_cv_time_interval_sec"],
        "description": "timing and frequency patterns"
    },
    "domain_characteristics": {
        "features": ["domain_cnt", "domain_concentration", "domain_entropy", "domain_diversity",
                    "suspicious_domain_ratio", "new_domain_ratio", "external_domain_ratio",
                    "suspicious_tld_ratio", "avg_subdomain_depth", "max_subdomain_depth",
                    "cross_domain_ratio", "referrer_diversity", "avg_domain_length",
                    "max_domain_length", "numeric_domain_ratio", "tld_diversity",
                    "internal_domain_count", "external_domain_count", "key_hostname_cnt",
                    "distinct_key_domain_count", "referer_domain_cnt"],
        "description": "domain communication patterns"
    },
    "http_behavior": {
        "features": ["http_status_cnt", "http_method_cnt", "error_ratio", "redirect_ratio",
                    "https_ratio", "http2_usage_ratio", "status_diversity", "method_diversity",
                    "http_version_diversity", "error_rate", "non_standard_method_ratio",
                    "protocol_consistency"],
        "description": "HTTP protocol behavior"
    },
    "content_types": {
        "features": ["req_content_type_cnt", "resp_content_type_cnt", "json_ratio", 
                    "html_ratio", "js_ratio", "executable_ratio", "script_ratio",
                    "css_ratio", "image_ratio", "xml_ratio", "content_type_diversity",
                    "content_type_mismatch_ratio", "compression_usage_ratio",
                    "avg_compression_ratio", "response_size_entropy", "json_response_ratio",
                    "html_response_ratio"],
        "description": "content type patterns"
    },
    "user_agent": {
        "features": ["ua_diversity", "ua_entropy", "bot_ratio", "suspicious_ua_ratio",
                    "ua_consistency", "automation_suspicion", "chrome_ratio", "firefox_ratio",
                    "safari_ratio"],
        "description": "user agent characteristics"
    },
    "response_characteristics": {
        "features": ["max_time_taken_ms", "avg_time_taken_ms", "std_time_taken_ms",
                    "response_size_cv", "large_response_ratio", "min_time_taken_ms",
                    "range_time_taken_ms", "median_time_taken_ms", "p25_time_taken_ms",
                    "p75_time_taken_ms", "p90_time_taken_ms", "p95_time_taken_ms",
                    "p99_time_taken_ms", "iqr_time_taken_ms", "cv_time_taken_ms",
                    "skewness_time_taken_ms", "kurtosis_time_taken_ms",
                    "outlier_ratio_time_taken_ms", "mad_time_taken_ms", "robust_cv_time_taken_ms"],
        "description": "server response patterns"
    },
    "network_topology": {
        "features": ["url_entropy", "avg_url_length", "avg_path_depth", "path_depth_variance",
                    "ip_diversity", "private_ip_ratio", "cdn_usage_ratio", "thirdparty_service_ratio",
                    "dependency_complexity"],
        "description": "network structure and routing patterns"
    },
    "security_indicators": {
        "features": ["mixed_content_risk", "cert_chain_depth_estimate", "secure_transport_ratio",
                    "referer_present_ratio", "same_origin_referer_ratio", "api_endpoint_ratio"],
        "description": "security posture indicators"
    },
    "sequence_patterns": {
        "features": ["sequence_num_keys", "sequence_max_key_length", "sequence_min_key_length",
                    "sequence_max_val", "sequence_min_val", "sequence_sum_val", "sequence_avg_val",
                    "sequence_std_val", "sequence_median_val", "sequence_range_val", "unique_actions"],
        "description": "behavioral sequence patterns"
    },
    "aggregate_metrics": {
        "features": ["transactions", "request_count", "range_timestamp", "cloud_traffic_pct",
                    "web_traffic_pct", "refered_traffic_pct"],
        "description": "overall traffic metrics"
    }
}

# Feature interpretation rules
FEATURE_INTERPRETATIONS = {
    # Volume-based features
    "sum_server_bytes": {
        "high_positive": "unusually high data download volume",
        "high_negative": "significantly lower data download than typical",
        "unit": "bytes"
    },
    "sum_client_bytes": {
        "high_positive": "unusually high data upload volume", 
        "high_negative": "significantly lower data upload than typical",
        "unit": "bytes"
    },
    
    # Time-based features
    "burst_ratio": {
        "high_positive": "concentrated burst activity pattern",
        "high_negative": "evenly distributed activity",
    },
    "night_activity_ratio": {
        "high_positive": "unusual nighttime activity",
        "high_negative": "normal daytime activity pattern",
    },
    "interval_regularity": {
        "high_positive": "highly regular/automated timing pattern",
        "high_negative": "irregular human-like timing",
    },
    
    # Domain features
    "suspicious_domain_ratio": {
        "high_positive": "high ratio of suspicious domains",
        "high_negative": "mostly trusted domains",
    },
    "new_domain_ratio": {
        "high_positive": "high ratio of newly registered domains",
        "high_negative": "established domains only",
    },
    "external_domain_ratio": {
        "high_positive": "high ratio of external domain communication",
        "high_negative": "mostly internal domain communication",
    },
    
    # HTTP features
    "error_ratio": {
        "high_positive": "high error rate indicating issues",
        "high_negative": "low error rate indicating normal operation",
    },
    "automation_suspicion": {
        "high_positive": "patterns suggesting automated/bot behavior",
        "high_negative": "normal human interaction patterns",
    },
    
    # Content features
    "executable_ratio": {
        "high_positive": "high ratio of executable content",
        "high_negative": "minimal executable content",
    },
    "script_ratio": {
        "high_positive": "high ratio of script content",
        "high_negative": "minimal script content", 
    },
    
    # Traffic volume statistics
    "avg_client_bytes": {
        "high_positive": "unusually high average upload size per request",
        "high_negative": "significantly lower average upload size than typical",
        "unit": "bytes"
    },
    "avg_server_bytes": {
        "high_positive": "unusually high average download size per response",
        "high_negative": "significantly lower average download size than typical",
        "unit": "bytes"
    },
    "total_data_volume": {
        "high_positive": "excessive total data transfer volume",
        "high_negative": "minimal data transfer activity",
        "unit": "bytes"
    },
    "avg_bytes_per_request": {
        "high_positive": "large average payload size per request",
        "high_negative": "small average payload size per request",
        "unit": "bytes"
    },
    "min_client_bytes": {
        "high_positive": "minimum upload size is unusually large",
        "high_negative": "minimum upload size is very small",
        "unit": "bytes"
    },
    "max_client_bytes": {
        "high_positive": "extreme maximum upload size detected",
        "high_negative": "maximum upload size is unusually small",
        "unit": "bytes"
    },
    "min_server_bytes": {
        "high_positive": "minimum download size is unusually large",
        "high_negative": "minimum download size is very small",
        "unit": "bytes"
    },
    "max_server_bytes": {
        "high_positive": "extreme maximum download size detected",
        "high_negative": "maximum download size is unusually small",
        "unit": "bytes"
    },
    "range_client_bytes": {
        "high_positive": "highly variable upload sizes",
        "high_negative": "consistent upload sizes",
        "unit": "bytes"
    },
    "range_server_bytes": {
        "high_positive": "highly variable download sizes",
        "high_negative": "consistent download sizes",
        "unit": "bytes"
    },
    "cv_client_bytes": {
        "high_positive": "high upload size variability relative to mean",
        "high_negative": "low upload size variability",
    },
    "cv_server_bytes": {
        "high_positive": "high download size variability relative to mean",
        "high_negative": "low download size variability",
    },
    "skewness_client_bytes": {
        "high_positive": "upload sizes heavily skewed toward large values",
        "high_negative": "upload sizes skewed toward small values",
    },
    "skewness_server_bytes": {
        "high_positive": "download sizes heavily skewed toward large values",
        "high_negative": "download sizes skewed toward small values",
    },
    "kurtosis_client_bytes": {
        "high_positive": "extreme outliers in upload sizes",
        "high_negative": "upload sizes clustered around mean",
    },
    "kurtosis_server_bytes": {
        "high_positive": "extreme outliers in download sizes",
        "high_negative": "download sizes clustered around mean",
    },
    "outlier_ratio_client_bytes": {
        "high_positive": "high proportion of anomalous upload sizes",
        "high_negative": "all upload sizes within normal range",
    },
    "outlier_ratio_server_bytes": {
        "high_positive": "high proportion of anomalous download sizes",
        "high_negative": "all download sizes within normal range",
    },
    "mad_client_bytes": {
        "high_positive": "high median absolute deviation in upload sizes",
        "high_negative": "consistent upload sizes around median",
        "unit": "bytes"
    },
    "mad_server_bytes": {
        "high_positive": "high median absolute deviation in download sizes",
        "high_negative": "consistent download sizes around median",
        "unit": "bytes"
    },
    
    # Time pattern features
    "avg_time_interval_sec": {
        "high_positive": "long delays between requests",
        "high_negative": "rapid-fire request pattern",
        "unit": "seconds"
    },
    "std_time_interval_sec": {
        "high_positive": "highly irregular request timing",
        "high_negative": "consistent request intervals",
        "unit": "seconds"
    },
    "median_time_interval_sec": {
        "high_positive": "typical long delays between requests",
        "high_negative": "typically rapid request pattern",
        "unit": "seconds"
    },
    "interval_entropy": {
        "high_positive": "chaotic/random timing patterns",
        "high_negative": "predictable timing patterns",
    },
    "hour_entropy": {
        "high_positive": "activity spread across all hours",
        "high_negative": "activity concentrated in specific hours",
    },
    "avg_interval_sec": {
        "high_positive": "long average delay between requests",
        "high_negative": "rapid average request rate",
        "unit": "seconds"
    },
    "peak_hour_concentration": {
        "high_positive": "activity highly concentrated in peak hour",
        "high_negative": "activity evenly distributed across hours",
    },
    "cv_time_interval_sec": {
        "high_positive": "highly variable request timing",
        "high_negative": "consistent request timing",
    },
    "skewness_time_interval_sec": {
        "high_positive": "timing skewed toward long delays",
        "high_negative": "timing skewed toward rapid requests",
    },
    "kurtosis_time_interval_sec": {
        "high_positive": "extreme timing outliers present",
        "high_negative": "timing clustered around average",
    },
    "outlier_ratio_time_interval_sec": {
        "high_positive": "many unusual timing intervals",
        "high_negative": "consistent timing patterns",
    },
    
    # Domain characteristics
    "domain_cnt": {
        "high_positive": "communication with many different domains",
        "high_negative": "communication with few domains",
    },
    "domain_concentration": {
        "high_positive": "requests concentrated on few domains",
        "high_negative": "requests distributed across many domains",
    },
    "domain_entropy": {
        "high_positive": "high randomness in domain names",
        "high_negative": "predictable domain name patterns",
    },
    "domain_diversity": {
        "high_positive": "wide variety of unique domains",
        "high_negative": "limited domain diversity",
    },
    "suspicious_tld_ratio": {
        "high_positive": "high ratio of suspicious top-level domains",
        "high_negative": "standard TLDs only",
    },
    "avg_subdomain_depth": {
        "high_positive": "deep subdomain nesting on average",
        "high_negative": "simple domain structure",
    },
    "max_subdomain_depth": {
        "high_positive": "extremely deep subdomain detected",
        "high_negative": "no deep subdomains",
    },
    "cross_domain_ratio": {
        "high_positive": "high cross-domain request activity",
        "high_negative": "mostly same-origin requests",
    },
    "referrer_diversity": {
        "high_positive": "requests from many different referrers",
        "high_negative": "consistent referrer sources",
    },
    "avg_domain_length": {
        "high_positive": "unusually long domain names on average",
        "high_negative": "short domain names",
    },
    "max_domain_length": {
        "high_positive": "extremely long domain name detected",
        "high_negative": "all domains have normal length",
    },
    "numeric_domain_ratio": {
        "high_positive": "high ratio of numeric/IP-based domains",
        "high_negative": "standard alphabetic domains",
    },
    "tld_diversity": {
        "high_positive": "wide variety of top-level domains",
        "high_negative": "limited TLD diversity",
    },
    "internal_domain_count": {
        "high_positive": "many internal domain communications",
        "high_negative": "few internal domains",
    },
    "external_domain_count": {
        "high_positive": "many external domain communications",
        "high_negative": "few external domains",
    },
    "key_hostname_cnt": {
        "high_positive": "many key hostnames identified",
        "high_negative": "few key hostnames",
    },
    "referer_domain_cnt": {
        "high_positive": "requests from many referrer domains",
        "high_negative": "limited referrer domains",
    },
    
    # HTTP behavior
    "http_status_cnt": {
        "high_positive": "wide variety of HTTP status codes",
        "high_negative": "limited status code variety",
    },
    "http_method_cnt": {
        "high_positive": "many different HTTP methods used",
        "high_negative": "limited HTTP methods",
    },
    "redirect_ratio": {
        "high_positive": "high proportion of redirect responses",
        "high_negative": "few or no redirects",
    },
    "https_ratio": {
        "high_positive": "high HTTPS usage",
        "high_negative": "mostly unencrypted HTTP",
    },
    "http2_usage_ratio": {
        "high_positive": "high HTTP/2 protocol usage",
        "high_negative": "legacy HTTP versions",
    },
    "status_diversity": {
        "high_positive": "many different response status codes",
        "high_negative": "consistent status codes",
    },
    "method_diversity": {
        "high_positive": "varied HTTP methods in use",
        "high_negative": "limited HTTP method variety",
    },
    "http_version_diversity": {
        "high_positive": "multiple HTTP protocol versions",
        "high_negative": "single HTTP version",
    },
    "error_rate": {
        "high_positive": "high proportion of error responses",
        "high_negative": "few errors encountered",
    },
    "non_standard_method_ratio": {
        "high_positive": "unusual HTTP methods detected",
        "high_negative": "standard HTTP methods only",
    },
    "protocol_consistency": {
        "high_positive": "consistent protocol usage",
        "high_negative": "mixed protocol usage",
    },
    
    # Content type features
    "req_content_type_cnt": {
        "high_positive": "many request content types used",
        "high_negative": "consistent request content types",
    },
    "resp_content_type_cnt": {
        "high_positive": "many response content types",
        "high_negative": "consistent response content types",
    },
    "json_ratio": {
        "high_positive": "high proportion of JSON responses",
        "high_negative": "minimal JSON content",
    },
    "html_ratio": {
        "high_positive": "high proportion of HTML responses",
        "high_negative": "minimal HTML content",
    },
    "js_ratio": {
        "high_positive": "high proportion of JavaScript responses",
        "high_negative": "minimal JavaScript content",
    },
    "css_ratio": {
        "high_positive": "high proportion of CSS responses",
        "high_negative": "minimal CSS content",
    },
    "image_ratio": {
        "high_positive": "high proportion of image responses",
        "high_negative": "minimal image content",
    },
    "xml_ratio": {
        "high_positive": "high proportion of XML responses",
        "high_negative": "minimal XML content",
    },
    "content_type_diversity": {
        "high_positive": "wide variety of content types",
        "high_negative": "limited content type variety",
    },
    "content_type_mismatch_ratio": {
        "high_positive": "high content type mismatches detected",
        "high_negative": "content types match expectations",
    },
    "compression_usage_ratio": {
        "high_positive": "high compression usage",
        "high_negative": "minimal compression used",
    },
    "avg_compression_ratio": {
        "high_positive": "high average compression efficiency",
        "high_negative": "poor compression efficiency",
    },
    "response_size_entropy": {
        "high_positive": "high variety in response sizes",
        "high_negative": "consistent response sizes",
    },
    "json_response_ratio": {
        "high_positive": "high proportion of JSON responses",
        "high_negative": "minimal JSON responses",
    },
    "html_response_ratio": {
        "high_positive": "high proportion of HTML responses",
        "high_negative": "minimal HTML responses",
    },
    
    # User agent features
    "ua_diversity": {
        "high_positive": "many different user agents",
        "high_negative": "consistent user agent usage",
    },
    "ua_entropy": {
        "high_positive": "high randomness in user agent strings",
        "high_negative": "predictable user agent patterns",
    },
    "bot_ratio": {
        "high_positive": "high proportion of bot/automated traffic",
        "high_negative": "minimal bot activity",
    },
    "suspicious_ua_ratio": {
        "high_positive": "high ratio of suspicious user agents",
        "high_negative": "normal user agent patterns",
    },
    "ua_consistency": {
        "high_positive": "consistent user agent usage",
        "high_negative": "highly varied user agents",
    },
    "chrome_ratio": {
        "high_positive": "high proportion of Chrome user agents",
        "high_negative": "minimal Chrome usage",
    },
    "firefox_ratio": {
        "high_positive": "high proportion of Firefox user agents",
        "high_negative": "minimal Firefox usage",
    },
    "safari_ratio": {
        "high_positive": "high proportion of Safari user agents",
        "high_negative": "minimal Safari usage",
    },
    
    # Response characteristics
    "max_time_taken_ms": {
        "high_positive": "extremely long maximum response time",
        "high_negative": "fast maximum response time",
        "unit": "milliseconds"
    },
    "avg_time_taken_ms": {
        "high_positive": "slow average response time",
        "high_negative": "fast average response time",
        "unit": "milliseconds"
    },
    "std_time_taken_ms": {
        "high_positive": "highly variable response times",
        "high_negative": "consistent response times",
        "unit": "milliseconds"
    },
    "min_time_taken_ms": {
        "high_positive": "slow minimum response time",
        "high_negative": "very fast minimum response time",
        "unit": "milliseconds"
    },
    "range_time_taken_ms": {
        "high_positive": "wide response time range",
        "high_negative": "narrow response time range",
        "unit": "milliseconds"
    },
    "median_time_taken_ms": {
        "high_positive": "slow median response time",
        "high_negative": "fast median response time",
        "unit": "milliseconds"
    },
    "cv_time_taken_ms": {
        "high_positive": "high response time variability",
        "high_negative": "consistent response times",
    },
    "skewness_time_taken_ms": {
        "high_positive": "response times skewed toward slow values",
        "high_negative": "response times skewed toward fast values",
    },
    "kurtosis_time_taken_ms": {
        "high_positive": "extreme response time outliers",
        "high_negative": "response times clustered around average",
    },
    "outlier_ratio_time_taken_ms": {
        "high_positive": "many unusual response times",
        "high_negative": "consistent response timing",
    },
    "mad_time_taken_ms": {
        "high_positive": "high response time deviation from median",
        "high_negative": "consistent response times around median",
        "unit": "milliseconds"
    },
    "robust_cv_time_taken_ms": {
        "high_positive": "robust high response time variability",
        "high_negative": "robust consistent response times",
    },
    
    # Network topology features
    "url_entropy": {
        "high_positive": "high randomness in URL patterns",
        "high_negative": "predictable URL structures",
    },
    "avg_url_length": {
        "high_positive": "unusually long URLs on average",
        "high_negative": "short URLs",
    },
    "avg_path_depth": {
        "high_positive": "deep URL path nesting",
        "high_negative": "shallow URL paths",
    },
    "path_depth_variance": {
        "high_positive": "highly variable URL path depths",
        "high_negative": "consistent URL path depths",
    },
    "ip_diversity": {
        "high_positive": "many different source IP addresses",
        "high_negative": "consistent source IPs",
    },
    "private_ip_ratio": {
        "high_positive": "high proportion of private IP addresses",
        "high_negative": "mostly public IP addresses",
    },
    "cdn_usage_ratio": {
        "high_positive": "high CDN usage",
        "high_negative": "minimal CDN usage",
    },
    "thirdparty_service_ratio": {
        "high_positive": "high third-party service usage",
        "high_negative": "minimal third-party services",
    },
    "dependency_complexity": {
        "high_positive": "high dependency complexity",
        "high_negative": "simple dependency structure",
    },
    
    # Security indicators
    "mixed_content_risk": {
        "high_positive": "high mixed content security risk",
        "high_negative": "no mixed content issues",
    },
    "cert_chain_depth_estimate": {
        "high_positive": "complex certificate chain structure",
        "high_negative": "simple certificate chain",
    },
    "secure_transport_ratio": {
        "high_positive": "high secure transport usage",
        "high_negative": "mostly insecure transport",
    },
    "referer_present_ratio": {
        "high_positive": "referrer headers usually present",
        "high_negative": "referrer headers often missing",
    },
    "same_origin_referer_ratio": {
        "high_positive": "high same-origin referrer usage",
        "high_negative": "mostly cross-origin referrers",
    },
    "api_endpoint_ratio": {
        "high_positive": "high API endpoint usage",
        "high_negative": "minimal API endpoint usage",
    },
    
    # Sequence patterns
    "sequence_num_keys": {
        "high_positive": "many different sequence patterns",
        "high_negative": "few sequence patterns",
    },
    "sequence_max_key_length": {
        "high_positive": "very long sequence patterns",
        "high_negative": "short sequence patterns",
    },
    "sequence_min_key_length": {
        "high_positive": "long minimum sequence patterns",
        "high_negative": "very short sequence patterns",
    },
    "sequence_max_val": {
        "high_positive": "high maximum sequence frequency",
        "high_negative": "low maximum sequence frequency",
    },
    "sequence_min_val": {
        "high_positive": "high minimum sequence frequency",
        "high_negative": "very low sequence frequencies",
    },
    "sequence_sum_val": {
        "high_positive": "high total sequence activity",
        "high_negative": "low total sequence activity",
    },
    "sequence_avg_val": {
        "high_positive": "high average sequence frequency",
        "high_negative": "low average sequence frequency",
    },
    "sequence_std_val": {
        "high_positive": "high sequence frequency variability",
        "high_negative": "consistent sequence frequencies",
    },
    "sequence_median_val": {
        "high_positive": "high median sequence frequency",
        "high_negative": "low median sequence frequency",
    },
    "sequence_range_val": {
        "high_positive": "wide sequence frequency range",
        "high_negative": "narrow sequence frequency range",
    },
    "unique_actions": {
        "high_positive": "many different action types",
        "high_negative": "few action types",
    },
    
    # Aggregate metrics
    "transactions": {
        "high_positive": "high transaction volume",
        "high_negative": "low transaction volume",
    },
    "request_count": {
        "high_positive": "high request count",
        "high_negative": "low request count",
    },
    "range_timestamp": {
        "high_positive": "activity over long time period",
        "high_negative": "activity over short time period",
        "unit": "seconds"
    },
    "cloud_traffic_pct": {
        "high_positive": "high proportion of cloud traffic",
        "high_negative": "minimal cloud traffic",
    },
    "web_traffic_pct": {
        "high_positive": "high proportion of web traffic",
        "high_negative": "minimal web traffic",
    },
    "refered_traffic_pct": {
        "high_positive": "high proportion of referred traffic",
        "high_negative": "minimal referred traffic",
    }
}


class ModelExplainer:
    """Generates human-readable explanations for model predictions using SHAP"""
    
    def __init__(self, model, feature_names: List[str], logger: Optional[logging.Logger] = None):
        """
        Initialize the ModelExplainer
        
        Args:
            model: The trained model (pipeline or estimator)
            feature_names: List of feature names
            logger: Optional logger instance
        """
        self.model = model
        self.feature_names = feature_names
        self.logger = logger or logging.getLogger(__name__)
        self.explainer = None
        
    def _get_base_estimator(self):
        """Extract the base estimator from a pipeline"""
        if hasattr(self.model, "named_steps"):
            # It's a pipeline - get the final classifier
            return self.model.named_steps["xgb"]
        else:
            return self.model
            
    def _create_explainer(self, features_scaled: NDArray):
        """Create SHAP explainer if not already created"""
        if self.explainer is None:
            base_estimator = self._get_base_estimator()
            self.explainer = shap.TreeExplainer(base_estimator)
            
    def calculate_shap_values(self, features_scaled: NDArray, 
                            observation_index: int,
                            predicted_class_index: int) -> Tuple[NDArray, float]:
        """
        Calculate SHAP values for a specific observation
        
        Returns:
            Tuple of (shap_values, expected_value)
        """
        self._create_explainer(features_scaled)
        
        # Get SHAP values for the specific observation
        chosen_instance = features_scaled[observation_index, :].reshape(1, -1)
        shap_values = self.explainer.shap_values(chosen_instance)

        # Handle multi-class output
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = self.explainer.expected_value[predicted_class_index]
            shap_values_for_class = shap_values[predicted_class_index]
        else:
            base_value = self.explainer.expected_value
            shap_values_for_class = shap_values

        return shap_values_for_class, base_value
        
    def get_top_features(self, shap_values: NDArray, top_n: int = None) -> List[Tuple[str, float, float]]:
        """
        Get top features by absolute SHAP value
        
        Returns:
            List of tuples (feature_name, shap_value, feature_value)
        """
        if top_n is None:
            top_n = constants.TOP_FEATURES_COUNT
            
        # Get absolute values for ranking
        abs_shap_values = np.abs(shap_values)
        top_indices = np.argsort(abs_shap_values)[0][-top_n:][::-1]
        
        top_features = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                try:
                    shap_value = shap_values[0][idx]
                except:
                    shap_value = shap_values[idx]
                top_features.append((feature_name, shap_value, idx))
                
        return top_features
        
    def _get_feature_group(self, feature_name: str) -> Optional[str]:
        """Get the feature group for a given feature"""
        for group_name, group_info in FEATURE_GROUPS.items():
            if feature_name in group_info["features"]:
                return group_name
        return None
        
    def _interpret_feature_impact(self, feature_name: str, shap_value: float, 
                                feature_value: float) -> str:
        """Generate human-readable interpretation for a feature's impact"""
        impact_direction = "increases" if shap_value > 0 else "decreases"
        
        if feature_name in FEATURE_INTERPRETATIONS:
            interp = FEATURE_INTERPRETATIONS[feature_name]
            
            # For features with units (bytes, milliseconds, etc.), check if the value makes sense
            if "unit" in interp and not np.isnan(feature_value):
                # Special handling for zero or near-zero values
                if abs(feature_value) < 1e-3:
                    if "bytes" in interp["unit"]:
                        interpretation = f"no data transfer detected"
                    elif "milliseconds" in interp["unit"] or "seconds" in interp["unit"]:
                        interpretation = f"instantaneous or missing response time"
                    else:
                        interpretation = f"zero {feature_name}"
                    context = f" ({feature_value:,.0f} {interp['unit']})"
                else:
                    # Use standard interpretation based on SHAP value direction
                    if shap_value > 0:
                        interpretation = interp.get("high_positive", f"high {feature_name}")
                    else:
                        interpretation = interp.get("high_negative", f"low {feature_name}")
                    context = f" ({feature_value:,.0f} {interp['unit']})"
            else:
                # For non-unit features, use SHAP value direction to determine interpretation
                if shap_value > 0:
                    interpretation = interp.get("high_positive", f"high {feature_name}")
                else:
                    interpretation = interp.get("high_negative", f"low {feature_name}")
                
                # Add quantitative context with just the value
                if not np.isnan(feature_value):
                    context = f" ({feature_value:.2f})"
                else:
                    context = ""
                
            return f"{interpretation}{context} {impact_direction} anomaly score"
        else:
            # Generic interpretation
            value_desc = "high" if feature_value > 0 else "low"
            return f"{value_desc} {feature_name} ({feature_value:.2f}) {impact_direction} anomaly score"
            
    def generate_text_explanation(self, 
                                features_scaled: NDArray,
                                observation_index: int,
                                observation_data: Dict[str, Any],
                                predicted_class: str,
                                predicted_proba: float,
                                top_n_features: int = None) -> str:
        """
        Generate a human-readable text explanation for the prediction
        
        Args:
            features_scaled: Scaled feature matrix
            observation_index: Index of the observation
            observation_data: Original observation data (with metadata)
            predicted_class: The predicted class name
            predicted_proba: Prediction probability
            top_n_features: Number of top features to explain
            
        Returns:
            Human-readable explanation string
        """

        if predicted_class == 0:
            return ""

        if top_n_features is None:
            top_n_features = constants.TOP_FEATURES_COUNT

        # Get application and domain info
        application = observation_data.get("application", "Unknown")
        domain = observation_data.get("domain", observation_data.get("key", "unknown domain"))
        
        # Calculate SHAP values
        # For binary classification, we explain the positive class (anomaly)
        predicted_class_index = 1
        shap_values, expected_value = self.calculate_shap_values(
            features_scaled, observation_index, predicted_class_index
        )

        # Get top contributing features
        top_features = self.get_top_features(shap_values, top_n_features)
        
        # Start building the explanation
        if predicted_proba >= 0.95:
            severity = "high confidence"
        elif predicted_proba >= 0.9:
            severity = "moderate confidence"
        else:
            severity = "low confidence"
            
        explanation_parts = [
            f"Communication from {application} to {domain} is flagged for potential supply chain compromise with {severity} (probability: {predicted_proba:.1%})."
        ]
        
        # Group features by category
        feature_groups = {}
        for feature_name, shap_value, feature_idx in top_features:
            group = self._get_feature_group(feature_name)
            if group not in feature_groups:
                feature_groups[group] = []
            
            # Get the actual feature value
            feature_value = features_scaled[observation_index, feature_idx]
            feature_groups[group].append((feature_name, shap_value, feature_value))
        
        # Build grouped explanations
        key_factors = []
        for group, features in feature_groups.items():
            if group and group in FEATURE_GROUPS:
                group_desc = FEATURE_GROUPS[group]["description"]
                group_factors = []
                
                for feature_name, shap_value, feature_value in features:
                    interpretation = self._interpret_feature_impact(
                        feature_name, shap_value, feature_value
                    )
                    group_factors.append(interpretation)
                
                if group_factors:
                    key_factors.append(f"{group_desc}: {', '.join(group_factors)}")
        
        # Add key factors to explanation
        if key_factors:
            explanation_parts.append("\nKey indicators:")
            for i, factor in enumerate(key_factors[:10], 1):  # Show top 10 groups
                explanation_parts.append(f"{i}. {factor}")
        
        # Add contextual summary
        if predicted_proba >= 0.95:
            explanation_parts.append(
                f"\nThis pattern significantly deviates from typical {application} behavior "
                f"and warrants immediate investigation."
            )
        elif predicted_proba >= 0.9:
            explanation_parts.append(
                f"\nThis pattern shows some deviation from typical {application} behavior "
                f"and should be monitored."
            )
            
        return "\n".join(explanation_parts)
        
    def save_shap_plot(self, 
                      features_scaled: NDArray,
                      observation_index: int,
                      predicted_class_index: int,
                      save_path: str,
                      max_display: int = 20):
        """
        Generate and save SHAP waterfall plot
        
        Args:
            features_scaled: Scaled feature matrix
            observation_index: Index of the observation
            predicted_class_index: Index of predicted class
            save_path: Path to save the plot
            max_display: Maximum features to display
        """
        # Calculate SHAP values
        shap_values, expected_value = self.calculate_shap_values(
            features_scaled, observation_index, predicted_class_index
        )
        
        # Create SHAP explanation object
        chosen_instance = features_scaled[observation_index, :]
        exp = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=chosen_instance,
            feature_names=self.feature_names
        )
        
        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        try:
            shap.waterfall_plot(exp[0], max_display=max_display, show=False)
        except IndexError:
            shap.waterfall_plot(exp, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"SHAP plot saved to {save_path}")