"""Features module"""

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
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
from scipy import stats

from .utils import load_json_file, save_json_data


def get_numeric_stats(events: List, field: str) -> Dict:
    """
    Calculate comprehensive statistical metrics for a numeric field in a set of events.
    Enhanced with percentiles, distribution shape, and robust statistics.

    Args:
        events (list): A list of event dictionaries containing the numeric field.
        field (str): The name of the numeric field to calculate metrics for.

    Returns:
        dict: A dictionary containing calculated metrics including basic stats,
              percentiles, skewness, kurtosis, and coefficient of variation.

    Raises:
        None
    """
    values = [e.get(field, 0) for e in events]

    if not values:
        return {}

    # Convert to numpy array for efficient computation
    values_array = np.array(values)

    # Basic statistics
    basic_stats = {
        f"avg_{field}": np.mean(values_array),
        f"std_{field}": np.std(values_array, ddof=1) if len(values) > 1 else 0,
        f"median_{field}": np.median(values_array),
        f"range_{field}": np.ptp(values_array),
        f"max_{field}": np.max(values_array),
        f"min_{field}": np.min(values_array),
        f"sum_{field}": np.sum(values_array),
    }

    # Percentile-based features (more robust with limited samples)
    percentiles = {
        f"p25_{field}": np.percentile(values_array, 25),
        f"p75_{field}": np.percentile(values_array, 75),
        f"p90_{field}": np.percentile(values_array, 90),
        f"p95_{field}": np.percentile(values_array, 95),
        f"p99_{field}": np.percentile(values_array, 99),
    }

    # Inter-quartile range (robust measure of spread)
    iqr = percentiles[f"p75_{field}"] - percentiles[f"p25_{field}"]
    percentiles[f"iqr_{field}"] = iqr

    # Coefficient of variation (normalized variance)
    mean_val = basic_stats[f"avg_{field}"]
    std_val = basic_stats[f"std_{field}"]
    cv = std_val / (abs(mean_val) + 1e-8) if mean_val != 0 else 0

    # Distribution shape measures
    shape_stats = {
        f"cv_{field}": cv,
        f"skewness_{field}": stats.skew(values_array) if len(values) > 2 else 0,
        f"kurtosis_{field}": stats.kurtosis(values_array) if len(values) > 3 else 0,
    }

    # Outlier detection features
    if iqr > 0:
        # Standard outlier bounds (1.5 * IQR rule)
        lower_bound = percentiles[f"p25_{field}"] - 1.5 * iqr
        upper_bound = percentiles[f"p75_{field}"] + 1.5 * iqr
        outliers = np.sum((values_array < lower_bound) | (values_array > upper_bound))
        outlier_ratio = outliers / len(values_array)
    else:
        outlier_ratio = 0

    outlier_stats = {
        f"outlier_ratio_{field}": outlier_ratio,
    }

    # Robust statistics (median-based alternatives)
    robust_stats = {
        f"mad_{field}": np.median(
            np.abs(values_array - np.median(values_array))
        ),  # Median Absolute Deviation
        f"robust_cv_{field}": np.median(np.abs(values_array - np.median(values_array)))
        / (np.median(values_array) + 1e-8),
    }

    # Combine all statistics
    all_stats = {}
    all_stats.update(basic_stats)
    all_stats.update(percentiles)
    all_stats.update(shape_stats)
    all_stats.update(outlier_stats)
    all_stats.update(robust_stats)

    # Convert numpy types to native Python types for JSON serialization
    for key, value in all_stats.items():
        if hasattr(value, "item"):  # numpy scalar
            all_stats[key] = value.item()
        elif isinstance(value, np.ndarray):
            all_stats[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            all_stats[key] = float(value)

    return all_stats


def log_inventory() -> None:
    """
    Placeholder function to support the ability to inventory applications running and their behavior.
    pass

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    # for k in summary:
    #     if k == 'key':
    #         logger.info(f"\t[x] Application = {summary[k]}")
    #     else:
    #         logger.info(f"\t\t[x] {k} = {summary[k]}")

    pass


def get_sequence_map(sequences: List[str]) -> Optional[Dict[str, int]]:
    """
    Generate a sequence map from a list of sequences.

    Args:
        sequences (List[str]): A list of sequences.

    Returns:
        Optional[Dict[str, int]]: A dictionary mapping sequences to their counts, or None if sequences is empty.
    """
    if not sequences:
        return None

    raw_result = dict(Counter(sequences))
    cut_off = 5
    result = dict()

    for k in raw_result:
        if int(raw_result[k]) >= cut_off:
            result[k] = raw_result[k]

    return result


def get_sequence_actions(actions: List[str]) -> List[str]:
    """
    Generate a list of unique action states from a list of actions.

    Args:
        actions (List[str]): A list of actions.

    Returns:
        List[str]: A list of unique action states.
    """
    action_states = []
    for a in actions:
        if not len(action_states) or action_states[-1] != a:
            action_states.append(a)
    return action_states


def potential_sequence(actions: List[str]) -> str:
    """
    Generate a potential sequence string from a list of actions.

    Args:
        actions (List[str]): A list of actions.

    Returns:
        str: A string representing the potential sequence of actions.
    """
    action_states = get_sequence_actions(actions)
    return " -> ".join(list(action_states))


def get_sequence_map_feature(
    sequence_map: Dict[str, int], feature: str
) -> Union[int, float]:
    """
    Extract a specific feature from a sequence map.

    Args:
        sequence_map (Dict[str, int]): A dictionary mapping sequences to their counts.
        feature (str): The feature to extract.

    Returns:
        Union[int, float]: The value of the specified feature.
    """
    key_lengths = [len(k.split("->")) for k in dict(sequence_map)]
    dist = [int(sequence_map[k]) for k in dict(sequence_map)]

    result = {
        "num_keys": len(sequence_map),
        "max_key_length": max(key_lengths) if key_lengths else 0,
        "min_key_length": min(key_lengths) if key_lengths else 0,
        "max_val": max(dist) if dist else 0,
        "min_val": min(dist) if dist else 0,
        "sum_val": sum(dist) if dist else 0,
        "avg_val": statistics.mean(dist) if dist else 0,
        "std_val": statistics.stdev(dist) if len(dist) > 1 else 0,
        "median_val": statistics.median(dist) if dist else 0,
        "range_val": (max(dist) - min(dist)) if dist else 0,
    }

    return result[feature]


def get_temporal_features(transactions: List[Dict]) -> Dict[str, float]:
    """
    Extract temporal pattern features from transaction data.

    Args:
        transactions (List[Dict]): List of transaction dictionaries sorted by timestamp

    Returns:
        Dict[str, float]: Dictionary of temporal features
    """
    if len(transactions) < 2:
        return {}

    timestamps = [t["timestamp"] for t in transactions]
    intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

    # Convert timestamps to hour of day for circadian analysis
    hours = [datetime.fromtimestamp(ts).hour for ts in timestamps]
    hour_counts = Counter(hours)

    # Request burst detection - count requests within 1-second windows
    burst_count = 0
    for i in range(len(intervals)):
        if intervals[i] < 1.0:  # Less than 1 second
            burst_count += 1

    # Entropy of inter-request times (regularity measure)
    if intervals:
        # Bin intervals into categories for entropy calculation
        interval_bins = []
        for interval in intervals:
            if interval < 0.1:
                interval_bins.append("very_fast")
            elif interval < 1.0:
                interval_bins.append("fast")
            elif interval < 10.0:
                interval_bins.append("normal")
            elif interval < 60.0:
                interval_bins.append("slow")
            else:
                interval_bins.append("very_slow")

        bin_counts = Counter(interval_bins)
        total = len(interval_bins)
        interval_entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in bin_counts.values()
            if count > 0
        )
    else:
        interval_entropy = 0.0

    # Off-hours activity (night: 11PM-6AM, weekend detection would need date info)
    night_requests = sum(1 for hour in hours if hour >= 23 or hour <= 6)
    night_activity_ratio = night_requests / len(hours) if hours else 0.0

    # Activity concentration (measure of temporal clustering)
    hour_entropy = 0.0
    if hour_counts:
        total_requests = sum(hour_counts.values())
        hour_entropy = -sum(
            (count / total_requests) * math.log2(count / total_requests)
            for count in hour_counts.values()
            if count > 0
        )

    try:
        interval_regularity = (
            (1.0 / (statistics.stdev(intervals) + 1e-6) if len(intervals) > 1 else 0.0)
            if intervals
            else 0.0
        )
    except statistics.StatisticsError:
        interval_regularity = 0.0

    return {
        "burst_ratio": burst_count / len(intervals) if intervals else 0.0,
        "interval_entropy": interval_entropy,
        "night_activity_ratio": night_activity_ratio,
        "hour_entropy": hour_entropy,
        "avg_interval_sec": statistics.mean(intervals) if intervals else 0.0,
        "interval_regularity": interval_regularity,
        "peak_hour_concentration": max(hour_counts.values()) / len(hours)
        if hour_counts
        else 0.0,
    }


def get_network_behavior_features(transactions: List[Dict]) -> Dict[str, float]:
    """
    Extract network behavior fingerprint features.

    Args:
        transactions (List[Dict]): List of transaction dictionaries

    Returns:
        Dict[str, float]: Dictionary of network behavior features
    """
    if not transactions:
        return {}

    # Redirect analysis
    redirect_statuses = ["301", "302", "303", "307", "308"]
    redirects = [
        t for t in transactions if t.get("http_status", "") in redirect_statuses
    ]
    redirect_ratio = len(redirects) / len(transactions)

    # Error rate patterns
    error_4xx = len(
        [t for t in transactions if t.get("http_status", "").startswith("4")]
    )
    error_5xx = len(
        [t for t in transactions if t.get("http_status", "").startswith("5")]
    )
    error_ratio = (error_4xx + error_5xx) / len(transactions)

    # URL structure analysis
    urls = [t.get("url", "") for t in transactions if t.get("url", "")]
    if urls:
        # Calculate URL entropy (measure of URL randomness)
        url_lengths = [len(url) for url in urls]
        url_chars = "".join(urls)
        char_counts = Counter(url_chars)
        total_chars = len(url_chars)
        url_entropy = (
            -sum(
                (count / total_chars) * math.log2(count / total_chars)
                for count in char_counts.values()
                if count > 0
            )
            if total_chars > 0
            else 0.0
        )

        # Path depth analysis
        path_depths = []
        for url in urls:
            try:
                parsed = urlparse(url)
                path_depth = len([p for p in parsed.path.split("/") if p])
                path_depths.append(path_depth)
            except:
                path_depths.append(0)

        avg_path_depth = statistics.mean(path_depths) if path_depths else 0.0
        try:
            path_depth_std = (
                statistics.stdev(path_depths) if len(path_depths) > 1 else 0.0
            )
        except statistics.StatisticsError:
            path_depth_std = 0.0
    else:
        url_entropy = 0.0
        avg_path_depth = 0.0
        path_depth_std = 0.0
        url_lengths = [0]

    # Response size consistency
    server_bytes = [t.get("server_bytes", 0) for t in transactions]
    if server_bytes and max(server_bytes) > 0 and len(server_bytes) > 1:
        try:
            size_cv = statistics.stdev(server_bytes) / (
                statistics.mean(server_bytes) + 1e-6
            )
        except statistics.StatisticsError:
            size_cv = 0.0
    else:
        size_cv = 0.0

    return {
        "redirect_ratio": redirect_ratio,
        "error_ratio": error_ratio,
        "url_entropy": url_entropy,
        "avg_url_length": statistics.mean(url_lengths) if url_lengths else 0.0,
        "avg_path_depth": avg_path_depth,
        "path_depth_variance": path_depth_std,
        "response_size_cv": size_cv,
        "status_diversity": len(set(t.get("http_status", "") for t in transactions)),
        "method_diversity": len(set(t.get("http_method", "") for t in transactions)),
    }


def get_content_analysis_features(transactions: List[Dict]) -> Dict[str, float]:
    """
    Extract content analysis features from transaction data.

    Args:
        transactions (List[Dict]): List of transaction dictionaries

    Returns:
        Dict[str, float]: Dictionary of content analysis features
    """
    if not transactions:
        return {}

    # Content type analysis
    req_types = []
    resp_types = []

    for t in transactions:
        if "req_content_type" in t and t["req_content_type"]:
            req_types.append(t["req_content_type"].split(";")[0].lower())
        if "resp_content_type" in t and t["resp_content_type"]:
            resp_types.append(t["resp_content_type"].split(";")[0].lower())

    # Web resource type ratios
    web_types = {
        "html": ["text/html"],
        "css": ["text/css"],
        "js": ["application/javascript", "text/javascript"],
        "image": ["image/jpeg", "image/png", "image/gif", "image/webp"],
        "json": ["application/json"],
        "xml": ["application/xml", "text/xml"],
    }

    type_counts = {}
    for category, mime_types in web_types.items():
        count = sum(1 for rt in resp_types if any(mt in rt for mt in mime_types))
        type_counts[category] = count

    total_typed = sum(type_counts.values())
    type_ratios = {
        f"{k}_ratio": (v / total_typed if total_typed > 0 else 0.0)
        for k, v in type_counts.items()
    }

    # Content type mismatches (security indicator)
    mismatches = 0
    for t in transactions:
        url = t.get("url", "")
        resp_type = t.get("resp_content_type", "").lower()

        # Check for suspicious mismatches
        if url.endswith(".js") and "javascript" not in resp_type:
            mismatches += 1
        elif url.endswith(".css") and "css" not in resp_type:
            mismatches += 1
        elif (
            url.endswith((".jpg", ".jpeg", ".png", ".gif")) and "image" not in resp_type
        ):
            mismatches += 1

    mismatch_ratio = mismatches / len(transactions)

    # Compression analysis
    compressed_responses = 0
    total_responses = 0
    compression_ratios = []

    for t in transactions:
        if "server_bytes" in t and "client_bytes" in t:
            server_bytes = t["server_bytes"]
            client_bytes = t["client_bytes"]

            if server_bytes > 0 and client_bytes > 0:
                compression_ratio = client_bytes / server_bytes
                compression_ratios.append(compression_ratio)
                total_responses += 1

                # Detect compressed content (ratio < 0.9 suggests compression)
                if compression_ratio < 0.9:
                    compressed_responses += 1

    compression_usage = (
        compressed_responses / total_responses if total_responses > 0 else 0.0
    )
    try:
        avg_compression = (
            statistics.mean(compression_ratios) if compression_ratios else 1.0
        )
    except statistics.StatisticsError:
        avg_compression = 1.0

    # Response size patterns
    server_bytes = [
        t.get("server_bytes", 0) for t in transactions if t.get("server_bytes", 0) > 0
    ]
    if server_bytes and len(server_bytes) > 0:
        # Detect unusually large responses (potential data exfiltration)
        try:
            size_threshold = (
                statistics.mean(server_bytes) + 2 * statistics.stdev(server_bytes)
                if len(server_bytes) > 1
                else statistics.mean(server_bytes)
            )
        except statistics.StatisticsError:
            size_threshold = statistics.mean(server_bytes) if server_bytes else 0
        large_responses = sum(1 for size in server_bytes if size > size_threshold)
        large_response_ratio = large_responses / len(server_bytes)

        # Size distribution entropy
        size_bins = []
        for size in server_bytes:
            if size < 1024:  # < 1KB
                size_bins.append("tiny")
            elif size < 10240:  # < 10KB
                size_bins.append("small")
            elif size < 102400:  # < 100KB
                size_bins.append("medium")
            elif size < 1048576:  # < 1MB
                size_bins.append("large")
            else:
                size_bins.append("huge")

        bin_counts = Counter(size_bins)
        total = len(size_bins)
        size_entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in bin_counts.values()
            if count > 0
        )
    else:
        large_response_ratio = 0.0
        size_entropy = 0.0

    features = {
        "content_type_mismatch_ratio": mismatch_ratio,
        "compression_usage_ratio": compression_usage,
        "avg_compression_ratio": avg_compression,
        "large_response_ratio": large_response_ratio,
        "response_size_entropy": size_entropy,
        "content_type_diversity": len(set(resp_types)),
    }

    # Add individual type ratios
    features.update(type_ratios)

    return features


def get_protocol_security_features(transactions: List[Dict]) -> Dict[str, float]:
    """
    Extract protocol-level security features from transaction data.

    Args:
        transactions (List[Dict]): List of transaction dictionaries

    Returns:
        Dict[str, float]: Dictionary of protocol security features
    """
    if not transactions:
        return {}

    # TLS/HTTPS analysis
    https_requests = sum(
        1 for t in transactions if t.get("uri_scheme", "").lower() == "https"
    )
    https_ratio = https_requests / len(transactions)

    # HTTP version analysis
    http_versions = [
        t.get("client_http_version", "")
        for t in transactions
        if t.get("client_http_version")
    ]
    version_diversity = len(set(http_versions))

    # HTTP/2 and modern protocol usage
    http2_requests = sum(1 for v in http_versions if "2" in v)
    http2_ratio = http2_requests / len(http_versions) if http_versions else 0.0

    # Security headers analysis (inferred from common patterns)
    security_headers = {
        "strict-transport-security": 0,
        "content-security-policy": 0,
        "x-frame-options": 0,
        "x-content-type-options": 0,
        "x-xss-protection": 0,
    }

    # Analyze response content types for security patterns
    resp_types = [t.get("resp_content_type", "").lower() for t in transactions]

    # Detect potential security header presence (heuristic based on response patterns)
    json_responses = sum(1 for rt in resp_types if "json" in rt)
    html_responses = sum(1 for rt in resp_types if "html" in rt)

    # Mixed content detection (HTTP resources on HTTPS pages)
    mixed_content_risk = 0
    if https_ratio > 0.5:  # If mostly HTTPS
        http_requests = len(transactions) - https_requests
        mixed_content_risk = http_requests / len(transactions)

    # Certificate chain depth estimation (heuristic based on subdomain complexity)
    domains = [t.get("domain", "") for t in transactions if t.get("domain")]
    cert_chain_depth_estimate = 0.0
    if domains:
        subdomain_levels = []
        for domain in domains:
            parts = domain.split(".")
            if len(parts) > 2:
                # Count subdomains as proxy for cert chain complexity
                subdomain_levels.append(len(parts) - 2)
            else:
                subdomain_levels.append(0)
        cert_chain_depth_estimate = (
            statistics.mean(subdomain_levels) if subdomain_levels else 0.0
        )

    # Protocol downgrade detection
    protocol_consistency = 1.0
    if https_ratio > 0 and https_ratio < 1.0:
        # Mixed protocols might indicate downgrade attacks
        protocol_consistency = (
            abs(https_ratio - 0.5) * 2
        )  # Closer to 0.5 = less consistent

    return {
        "https_ratio": https_ratio,
        "http_version_diversity": version_diversity,
        "http2_usage_ratio": http2_ratio,
        "mixed_content_risk": mixed_content_risk,
        "cert_chain_depth_estimate": cert_chain_depth_estimate,
        "protocol_consistency": protocol_consistency,
        "json_response_ratio": json_responses / len(resp_types) if resp_types else 0.0,
        "html_response_ratio": html_responses / len(resp_types) if resp_types else 0.0,
        "secure_transport_ratio": https_ratio,
    }


def get_header_fingerprint_features(transactions: List[Dict]) -> Dict[str, float]:
    """
    Extract HTTP header fingerprinting features from transaction data.

    Args:
        transactions (List[Dict]): List of transaction dictionaries

    Returns:
        Dict[str, float]: Dictionary of header fingerprint features
    """
    if not transactions:
        return {}

    user_agents = [t.get("useragent", "") for t in transactions if t.get("useragent")]

    # User-Agent analysis
    ua_diversity = len(set(user_agents))
    ua_consistency = 1.0 - (ua_diversity / len(user_agents)) if user_agents else 0.0

    # User-Agent entropy (measure of randomness)
    if user_agents:
        all_ua_chars = "".join(user_agents)
        char_counts = Counter(all_ua_chars)
        total_chars = len(all_ua_chars)
        ua_entropy = (
            -sum(
                (count / total_chars) * math.log2(count / total_chars)
                for count in char_counts.values()
                if count > 0
            )
            if total_chars > 0
            else 0.0
        )
    else:
        ua_entropy = 0.0

    # Common User-Agent patterns
    browser_patterns = {
        "chrome": sum(1 for ua in user_agents if "chrome" in ua.lower()),
        "firefox": sum(1 for ua in user_agents if "firefox" in ua.lower()),
        "safari": sum(1 for ua in user_agents if "safari" in ua.lower()),
        "edge": sum(1 for ua in user_agents if "edge" in ua.lower()),
    }

    # Bot/automated tool detection
    bot_indicators = [
        "bot",
        "crawler",
        "spider",
        "scraper",
        "automated",
        "python",
        "curl",
        "wget",
    ]
    bot_requests = sum(
        1
        for ua in user_agents
        if any(indicator in ua.lower() for indicator in bot_indicators)
    )
    bot_ratio = bot_requests / len(user_agents) if user_agents else 0.0

    # Suspicious User-Agent characteristics
    suspicious_ua_count = 0
    for ua in user_agents:
        # Very short or very long UAs are suspicious
        if len(ua) < 10 or len(ua) > 500:
            suspicious_ua_count += 1
        # Unusual character patterns
        if re.search(r"[^\w\s\-\.\(\)\/;:,]", ua):
            suspicious_ua_count += 1

    suspicious_ua_ratio = suspicious_ua_count / len(user_agents) if user_agents else 0.0

    # Referer header analysis
    referers = [t.get("referer", "") for t in transactions if t.get("referer")]
    referer_present_ratio = len(referers) / len(transactions)

    # Referer consistency (same-origin vs cross-origin)
    same_origin_referers = 0
    for t in transactions:
        domain = t.get("domain", "")
        referer = t.get("referer", "")
        if referer and domain:
            try:
                referer_domain = urlparse(referer).netloc
                if referer_domain == domain:
                    same_origin_referers += 1
            except:
                pass

    same_origin_referer_ratio = (
        same_origin_referers / len(referers) if referers else 0.0
    )

    # Content-Type consistency
    req_types = [
        t.get("req_content_type", "") for t in transactions if t.get("req_content_type")
    ]
    resp_types = [
        t.get("resp_content_type", "")
        for t in transactions
        if t.get("resp_content_type")
    ]

    content_type_diversity = len(set(req_types + resp_types))

    return {
        "ua_diversity": ua_diversity,
        "ua_consistency": ua_consistency,
        "ua_entropy": ua_entropy,
        "bot_ratio": bot_ratio,
        "suspicious_ua_ratio": suspicious_ua_ratio,
        "referer_present_ratio": referer_present_ratio,
        "same_origin_referer_ratio": same_origin_referer_ratio,
        "content_type_diversity": content_type_diversity,
        "chrome_ratio": browser_patterns["chrome"] / len(user_agents)
        if user_agents
        else 0.0,
        "firefox_ratio": browser_patterns["firefox"] / len(user_agents)
        if user_agents
        else 0.0,
        "safari_ratio": browser_patterns["safari"] / len(user_agents)
        if user_agents
        else 0.0,
    }


def get_supply_chain_indicators(transactions: List[Dict]) -> Dict[str, float]:
    """
    Extract supply chain specific security indicators from transaction data.

    Args:
        transactions (List[Dict]): List of transaction dictionaries

    Returns:
        Dict[str, float]: Dictionary of supply chain security features
    """
    if not transactions:
        return {}

    domains = [t.get("domain", "") for t in transactions if t.get("domain")]
    urls = [t.get("url", "") for t in transactions if t.get("url")]

    # Dependency tracking - identify external vs internal domains
    unique_domains = set(domains)

    # Classify domains as internal vs external (heuristic)
    internal_domains = set()
    external_domains = set()

    for domain in unique_domains:
        # Heuristics for internal vs external
        if any(
            indicator in domain.lower()
            for indicator in ["localhost", "127.0.0.1", "internal", "corp", "local"]
        ):
            internal_domains.add(domain)
        else:
            external_domains.add(domain)

    external_domain_ratio = (
        len(external_domains) / len(unique_domains) if unique_domains else 0.0
    )

    # CDN and third-party service detection
    cdn_domains = set()
    thirdparty_services = set()

    cdn_indicators = [
        "cdn",
        "cloudfront",
        "akamai",
        "fastly",
        "cloudflare",
        "keycdn",
        "maxcdn",
    ]
    service_indicators = [
        "googleapis",
        "facebook",
        "twitter",
        "linkedin",
        "analytics",
        "tracking",
        "ads",
        "doubleclick",
        "googlesyndication",
    ]

    for domain in unique_domains:
        domain_lower = domain.lower()
        if any(indicator in domain_lower for indicator in cdn_indicators):
            cdn_domains.add(domain)
        if any(indicator in domain_lower for indicator in service_indicators):
            thirdparty_services.add(domain)

    cdn_usage_ratio = len(cdn_domains) / len(unique_domains) if unique_domains else 0.0
    thirdparty_service_ratio = (
        len(thirdparty_services) / len(unique_domains) if unique_domains else 0.0
    )

    # Typosquatting detection
    suspicious_domains = 0
    for domain in unique_domains:
        # Check for suspicious patterns
        if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):  # IP addresses
            suspicious_domains += 1
        elif len(domain.split(".")) > 5:  # Very deep subdomains
            suspicious_domains += 1
        elif re.search(r"[0-9]{5,}", domain):  # Long numeric sequences
            suspicious_domains += 1
        elif (
            any(char in domain for char in ["-", "_"])
            and domain.count("-") + domain.count("_") > 3
        ):
            suspicious_domains += 1  # Too many hyphens/underscores

    suspicious_domain_ratio = (
        suspicious_domains / len(unique_domains) if unique_domains else 0.0
    )

    # Domain age estimation (heuristic based on patterns)
    new_domain_indicators = 0
    for domain in unique_domains:
        # Heuristics for potentially new/suspicious domains
        if re.search(r"[a-z]{8,}\.tk|\.ml|\.ga|\.cf", domain.lower()):  # Free TLDs
            new_domain_indicators += 1
        elif re.search(r"[0-9]+[a-z]+[0-9]+", domain):  # Mixed alphanumeric
            new_domain_indicators += 1

    new_domain_ratio = (
        new_domain_indicators / len(unique_domains) if unique_domains else 0.0
    )

    # Request pattern analysis for supply chain attacks
    api_endpoints = sum(1 for url in urls if "/api/" in url.lower())
    api_endpoint_ratio = api_endpoints / len(urls) if urls else 0.0

    # File type analysis for potential payload delivery
    file_extensions = []
    for url in urls:
        path = urlparse(url).path
        if "." in path:
            ext = path.split(".")[-1].lower()
            if len(ext) <= 5:  # Reasonable extension length
                file_extensions.append(ext)

    executable_extensions = ["exe", "dll", "bin", "com", "scr", "bat", "cmd", "ps1"]
    script_extensions = ["js", "vbs", "php", "py", "sh", "pl"]

    executable_requests = sum(
        1 for ext in file_extensions if ext in executable_extensions
    )
    script_requests = sum(1 for ext in file_extensions if ext in script_extensions)

    executable_ratio = (
        executable_requests / len(file_extensions) if file_extensions else 0.0
    )
    script_ratio = script_requests / len(file_extensions) if file_extensions else 0.0

    # Time-based anomaly detection
    timestamps = [t.get("timestamp", 0) for t in transactions if t.get("timestamp")]
    if len(timestamps) > 1:
        time_intervals = [
            timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))
        ]
        # Detect very regular intervals (possible automated attacks)
        if time_intervals and len(time_intervals) > 1:
            try:
                interval_cv = statistics.stdev(time_intervals) / (
                    statistics.mean(time_intervals) + 1e-6
                )
                automation_suspicion = (
                    1.0 / (interval_cv + 1e-6) if interval_cv < 0.1 else 0.0
                )
            except statistics.StatisticsError:
                automation_suspicion = 0.0
        else:
            automation_suspicion = 0.0
    else:
        automation_suspicion = 0.0

    return {
        "external_domain_ratio": external_domain_ratio,
        "cdn_usage_ratio": cdn_usage_ratio,
        "thirdparty_service_ratio": thirdparty_service_ratio,
        "suspicious_domain_ratio": suspicious_domain_ratio,
        "new_domain_ratio": new_domain_ratio,
        "api_endpoint_ratio": api_endpoint_ratio,
        "executable_ratio": executable_ratio,
        "script_ratio": script_ratio,
        "automation_suspicion": automation_suspicion,
        "dependency_complexity": len(unique_domains),
        "internal_domain_count": len(internal_domains),
        "external_domain_count": len(external_domains),
    }


def get_behavioral_baseline_features(transactions: List[Dict]) -> Dict[str, float]:
    """
    Extract behavioral baseline features for anomaly detection.

    Args:
        transactions (List[Dict]): List of transaction dictionaries

    Returns:
        Dict[str, float]: Dictionary of behavioral baseline features
    """
    if not transactions:
        return {}

    # Geographic consistency (based on IP patterns)
    src_ips = [t.get("src_ip", "") for t in transactions if t.get("src_ip")]
    unique_ips = set(src_ips)
    ip_diversity = len(unique_ips)

    # Private IP address detection
    private_ip_count = 0
    for ip in unique_ips:
        if ip.startswith(("10.", "172.", "192.168.", "127.")):
            private_ip_count += 1

    private_ip_ratio = private_ip_count / len(unique_ips) if unique_ips else 0.0

    # Request volume patterns
    total_bytes = sum(
        t.get("client_bytes", 0) + t.get("server_bytes", 0) for t in transactions
    )
    avg_bytes_per_request = total_bytes / len(transactions) if transactions else 0.0

    # Error pattern analysis
    error_statuses = ["4", "5"]  # 4xx and 5xx errors
    error_requests = sum(
        1
        for t in transactions
        if any(t.get("http_status", "").startswith(status) for status in error_statuses)
    )
    error_rate = error_requests / len(transactions)

    # Request method diversity
    methods = [t.get("http_method", "") for t in transactions if t.get("http_method")]
    method_diversity = len(set(methods))

    # Non-standard methods
    standard_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
    non_standard_methods = sum(
        1 for method in methods if method not in standard_methods
    )
    non_standard_method_ratio = non_standard_methods / len(methods) if methods else 0.0

    return {
        "ip_diversity": ip_diversity,
        "private_ip_ratio": private_ip_ratio,
        "avg_bytes_per_request": avg_bytes_per_request,
        "error_rate": error_rate,
        "method_diversity": method_diversity,
        "non_standard_method_ratio": non_standard_method_ratio,
        "total_data_volume": total_bytes,
        "request_count": len(transactions),
    }


def get_graph_based_features(transactions: List[Dict]) -> Dict[str, float]:
    """
    Extract graph-based domain relationship features.

    Args:
        transactions (List[Dict]): List of transaction dictionaries

    Returns:
        Dict[str, float]: Dictionary of graph-based features
    """
    if not transactions:
        return {}

    # Extract domains and subdomains
    domains = []
    subdomains = []
    tlds = []

    for t in transactions:
        domain = t.get("domain", "")
        if domain:
            domains.append(domain)

            # Parse domain structure
            parts = domain.split(".")
            if len(parts) >= 2:
                tlds.append(parts[-1])
                if len(parts) > 2:
                    # Count subdomains (everything before main domain)
                    subdomain_count = len(parts) - 2
                    subdomains.append(subdomain_count)
                else:
                    subdomains.append(0)

    if not domains:
        return {}

    # Domain diversity and concentration
    unique_domains = set(domains)
    domain_concentration = len(domains) / len(unique_domains) if unique_domains else 0

    # TLD analysis
    tld_counts = Counter(tlds)
    suspicious_tlds = {".tk", ".ml", ".ga", ".cf", ".bit", ".onion"}
    suspicious_tld_count = sum(1 for tld in tlds if tld in suspicious_tlds)
    suspicious_tld_ratio = suspicious_tld_count / len(tlds) if tlds else 0.0

    # Subdomain complexity
    try:
        avg_subdomain_depth = statistics.mean(subdomains) if subdomains else 0.0
        max_subdomain_depth = max(subdomains) if subdomains else 0
    except (statistics.StatisticsError, ValueError):
        avg_subdomain_depth = 0.0
        max_subdomain_depth = 0

    # Cross-domain request patterns
    referrers = []
    cross_domain_requests = 0

    for t in transactions:
        domain = t.get("domain", "")
        referrer = t.get("referrer", "") or t.get("referer", "")

        if referrer and domain:
            try:
                referrer_domain = urlparse(referrer).netloc
                referrers.append(referrer_domain)

                # Check if referrer domain differs from request domain
                if referrer_domain != domain and referrer_domain:
                    cross_domain_requests += 1
            except:
                pass

    cross_domain_ratio = cross_domain_requests / len(transactions)
    referrer_diversity = len(set(referrers)) if referrers else 0

    # Domain name entropy (randomness measure)
    all_domain_chars = "".join(domains)
    if all_domain_chars:
        char_counts = Counter(all_domain_chars)
        total_chars = len(all_domain_chars)
        domain_entropy = -sum(
            (count / total_chars) * math.log2(count / total_chars)
            for count in char_counts.values()
            if count > 0
        )
    else:
        domain_entropy = 0.0

    # Domain length analysis (longer domains often suspicious)
    domain_lengths = [len(d) for d in domains]
    try:
        avg_domain_length = statistics.mean(domain_lengths) if domain_lengths else 0.0
        max_domain_length = max(domain_lengths) if domain_lengths else 0
    except (statistics.StatisticsError, ValueError):
        avg_domain_length = 0.0
        max_domain_length = 0

    # Numeric domain detection (IP addresses, suspicious patterns)
    numeric_domains = 0
    for domain in unique_domains:
        # Check for IP addresses or domains with many numbers
        if any(char.isdigit() for char in domain):
            digit_ratio = sum(1 for char in domain if char.isdigit()) / len(domain)
            if digit_ratio > 0.3:  # More than 30% digits
                numeric_domains += 1

    numeric_domain_ratio = (
        numeric_domains / len(unique_domains) if unique_domains else 0.0
    )

    # Resource loading patterns (CDN detection)
    cdn_indicators = {"cdn", "static", "assets", "media", "img", "js", "css"}
    cdn_requests = sum(
        1
        for domain in domains
        if any(indicator in domain.lower() for indicator in cdn_indicators)
    )
    cdn_usage_ratio = cdn_requests / len(domains)

    return {
        "domain_concentration": domain_concentration,
        "domain_diversity": len(unique_domains),
        "suspicious_tld_ratio": suspicious_tld_ratio,
        "avg_subdomain_depth": avg_subdomain_depth,
        "max_subdomain_depth": max_subdomain_depth,
        "cross_domain_ratio": cross_domain_ratio,
        "referrer_diversity": referrer_diversity,
        "domain_entropy": domain_entropy,
        "avg_domain_length": avg_domain_length,
        "max_domain_length": max_domain_length,
        "numeric_domain_ratio": numeric_domain_ratio,
        "cdn_usage_ratio": cdn_usage_ratio,
        "tld_diversity": len(set(tlds)),
    }


def grab_application_summary(traffic_map: Dict, key: str, fields: list) -> Dict:
    """
    Generate a summary of application traffic based on the provided traffic map and key.

    Args:
        traffic_map (dict): A dictionary mapping application keys to lists of transaction events.
        key (str): The key identifying the application in the traffic map.
        fields (list): The keys to aggregate the traffic over

    Returns:
        dict: A dictionary containing a summary of the application's traffic, including metrics
        such as transaction count, distinct domain count, HTTP methods, status codes, and more.

    Raises:
        None
    """

    transactions = sorted(traffic_map[key], key=lambda x: x["timestamp"])
    sequence_window_length = 5

    for i, transaction in enumerate(transactions):
        if i >= 1:
            transaction["time_interval_sec"] = (
                transaction["timestamp"] - transactions[i - 1]["timestamp"]
            )
        else:
            transaction["time_interval_sec"] = 0.0

        actions = [
            transactions[i - d]["action"]
            for d in range(sequence_window_length)
            if (i - d) >= 0
        ]
        transaction["potential_sequence"] = potential_sequence(actions)

    transaction_count = len(transactions)
    seq = get_sequence_map(t["potential_sequence"] for t in transactions)
    distinct_domains = set(
        t.get("domain", "") for t in transactions if t.get("domain", "")
    )
    distinct_domain_count = len(distinct_domains)
    distinct_referrer_count = len(
        set(
            urlparse(t.get("referrer", "")).netloc
            for t in transactions
            if t.get("referrer", "")
        )
    )
    distinct_http_methods = sorted(
        list(
            set(
                t.get("http_method", "")
                for t in transactions
                if t.get("http_method", "")
            )
        )
    )
    distinct_http_method_count = len(distinct_http_methods)
    distinct_statuses = sorted(
        list(
            set(
                t.get("http_status", "")
                for t in transactions
                if t.get("http_status", "")
            )
        )
    )
    distinct_status_count = len(distinct_statuses)
    distinct_http_method_count = len(distinct_http_methods)

    distinct_req_types = sorted(
        list(
            set(
                req_type.split(";")[0]
                for t in transactions
                for req_type in t.get("req_types", [])
            )
        )
    )
    distinct_req_type_count = len(distinct_req_types)

    distinct_resp_types = sorted(
        list(
            set(
                resp_type.split(";")[0]
                for t in transactions
                for resp_type in t.get("resp_types", [])
            )
        )
    )
    distinct_resp_type_count = len(distinct_resp_types)

    referred_traffic_percent = (
        len([1 for t in transactions if "referrer" in t and t["referrer"]]) * 100.0
    ) / transaction_count
    cloud_traffic_percent = (
        len(
            [
                1
                for t in transactions
                if "traffic_type" in t and t["traffic_type"] == "cloud"
            ]
        )
        * 100.0
    ) / transaction_count
    web_traffic_percent = (
        len(
            [
                1
                for t in transactions
                if "traffic_type" in t and t["traffic_type"] == "non_cloud"
            ]
        )
        * 100.0
    ) / transaction_count

    key_domains = sorted(
        list(set(h for t in transactions for h in t.get("key_hostnames", [])))
    )
    distinct_key_domain_count = len(key_domains)
    range_timestamp = int(transactions[-1]["timestamp"] - transactions[0]["timestamp"])
    time_taken_stats = get_numeric_stats(events=transactions, field="time_taken_ms")
    client_byte_stats = get_numeric_stats(events=transactions, field="client_bytes")
    server_byte_stats = get_numeric_stats(events=transactions, field="server_bytes")
    time_interval_sec_stats = get_numeric_stats(
        events=transactions[1:], field="time_interval_sec"
    )
    applications = list(
        set(t.get("application", "") for t in transactions if t.get("application", ""))
    )
    unique_actions = set(t["action"] for t in transactions if "action" in t)

    if len(applications) > 1:
        raise Exception(
            "[!] More than one application detected for the same useragent", key
        )

    summary = {
        "key": key,
        "application": applications[0],
        "transactions": transaction_count,
        "refered_traffic_pct": referred_traffic_percent,
        "referer_domain_cnt": distinct_referrer_count,
        "unique_actions": len(unique_actions),
        "sequence_num_keys": get_sequence_map_feature(seq, "num_keys"),
        "sequence_max_key_length": get_sequence_map_feature(seq, "max_key_length"),
        "sequence_min_key_length": get_sequence_map_feature(seq, "min_key_length"),
        "sequence_max_val": get_sequence_map_feature(seq, "max_val"),
        "sequence_min_val": get_sequence_map_feature(seq, "min_val"),
        "sequence_sum_val": get_sequence_map_feature(seq, "sum_val"),
        "sequence_avg_val": get_sequence_map_feature(seq, "avg_val"),
        "sequence_std_val": get_sequence_map_feature(seq, "std_val"),
        "sequence_median_val": get_sequence_map_feature(seq, "median_val"),
        "sequence_range_val": get_sequence_map_feature(seq, "range_val"),
        "http_status_cnt": distinct_status_count,
        "http_method_cnt": distinct_http_method_count,
        "req_content_type_cnt": distinct_req_type_count,
        "resp_content_type_cnt": distinct_resp_type_count,
        "range_timestamp": range_timestamp,
        "cloud_traffic_pct": cloud_traffic_percent,
        "web_traffic_pct": web_traffic_percent,
        "http_methods": distinct_http_methods,
        "http_statuses": distinct_statuses,
        "req_content_types": distinct_req_types,
        "resp_content_types": distinct_resp_types,
        "domain_cnt": distinct_domain_count,
        "key_hostnames": key_domains,
        "key_hostname_cnt": distinct_key_domain_count,
        "distinct_key_domain_count": distinct_key_domain_count,
        # TODO: Add this back
        # 'distinct_url_count': distinct_url_count,
        # 'distinct_version_count': distinct_version_count,
        # 'distinct_file_count': distinct_file_count,
        # 'distinct_file_types': distinct_file_types,
        # 'distinct_file_type_count': distinct_file_type_count,
        # 'distinct_service_count': distinct_service_count,
    }

    if "domain" in fields:
        summary.update(
            {"domain": list(distinct_domains)[0] if distinct_domains else ""}
        )

    summary.update(time_taken_stats)
    summary.update(client_byte_stats)
    summary.update(server_byte_stats)
    summary.update(time_interval_sec_stats)

    # Add enhanced feature extraction
    temporal_features = get_temporal_features(transactions)
    network_features = get_network_behavior_features(transactions)
    content_features = get_content_analysis_features(transactions)
    graph_features = get_graph_based_features(transactions)

    # Add new security-focused features
    protocol_security_features = get_protocol_security_features(transactions)
    header_fingerprint_features = get_header_fingerprint_features(transactions)
    supply_chain_features = get_supply_chain_indicators(transactions)
    behavioral_features = get_behavioral_baseline_features(transactions)

    summary.update(temporal_features)
    summary.update(network_features)
    summary.update(content_features)
    summary.update(graph_features)
    summary.update(protocol_security_features)
    summary.update(header_fingerprint_features)
    summary.update(supply_chain_features)
    summary.update(behavioral_features)

    return summary


def aggregate_app_traffic(
    fields: List[str], input_path: str, output_path: str, min_transactions: int
) -> None:
    """
    Aggregate application traffic data from enriched events and save the summaries to a JSON file.

    Args:
        fields (List[str]): The field in the events to use as the application identifier.
        input_path (str): The path to the input JSON file containing enriched events.
        output_path (str): The path to the output JSON file where summaries will be saved.
        min_transactions (int): The minimum number of transactions we need to make a judgement call.

    Returns:
        None

    Raises:
        None
    """
    logger = logging.getLogger(__name__)
    traffic_map = dict()
    enriched_events = load_json_file(input_path)

    for e in enriched_events:
        # TODO: Handle cases where the app_field / user agent is empty
        if all(field in e for field in fields):
            field_key = " - ".join([e[field] for field in fields])
            key = f"{field_key} on {e['src_ip']}"
            if key in traffic_map:
                traffic_map[key].append(e)
            else:
                traffic_map[key] = [e]

    logger.info(
        f"[x] Found a total of {len(traffic_map)} distinct {fields} values in provided event file"
    )

    summaries = []
    for key in traffic_map:
        summary = grab_application_summary(
            traffic_map=traffic_map, key=key, fields=fields
        )
        if summary["transactions"] > min_transactions:
            summaries.append(summary)
        else:
            # TODO: Format the applications that did not have enough traffic in a nicer way
            # logger.info(f"\t[x] Application = {summary['key']}
            # (Number of transactions {summary['transactions']} did not meet the minimum required amount)")
            pass

    save_json_data(summaries, output_path)
