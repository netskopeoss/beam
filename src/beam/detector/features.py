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
import statistics
from collections import Counter
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

from .utils import load_json_file, save_json_data


def get_numeric_stats(events: List, field: str) -> Dict:
    """
    Calculate key metrics for a numeric field in a set of events.

    Args:
        events (list): A list of event dictionaries containing the numeric field.
        field (str): The name of the numeric field to calculate metrics for.

    Returns:
        dict: A dictionary containing calculated metrics such as average, standard deviation,
              median, range, max, min, and sum of the field values.

    Raises:
        None
    """
    values = [e.get(field, 0) for e in events]

    # distinct_domains = sorted(list({e['domain'] for e in events}))

    # domain_std_values = []
    # for d in distinct_domains:
    #     values_for_domain = []
    #     for e in events:
    #         if e['domain'] == d:
    #             values_for_domain.append(e.get(field, 0))
    #     domain_std_values.append(statistics.stdev(values_for_domain) if len(values_for_domain) > 1 else 0)

    return {
        # f'median_std_{field}': statistics.median(domain_std_values) if len(values) > 0 else 0,
        f"avg_{field}": statistics.mean(values) if len(values) > 0 else 0,
        f"std_{field}": statistics.stdev(values) if len(values) > 1 else 0,
        f"median_{field}": statistics.median(values) if len(values) > 0 else 0,
        f"range_{field}": max(values) - min(values) if len(values) > 0 else 0,
        f"max_{field}": max(values) if len(values) > 0 else 0,
        f"min_{field}": min(values) if len(values) > 0 else 0,
        f"sum_{field}": sum(values) if len(values) > 0 else 0,
    }


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


def grab_application_summary(traffic_map: Dict, key: str) -> Dict:
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
            transaction["time_interval_sec"] = (transaction["timestamp"] - transactions[i - 1]["timestamp"])
        else:
            transaction["time_interval_sec"] = 0.0

        actions = [transactions[i - d]["action"] for d in range(sequence_window_length) if (i - d) >= 0]
        transaction["potential_sequence"] = potential_sequence(actions)

    transaction_count = len(transactions)
    seq = get_sequence_map(t["potential_sequence"] for t in transactions)
    distinct_domains = set(t.get("domain", "") for t in transactions if t.get("domain", ""))
    distinct_domain_count = len(distinct_domains)
    distinct_url_count = len(set(t.get("url", "") for t in transactions if t.get("url", "")))
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
    distinct_version_count = len(
        set(
            t.get("client_http_version", "")
            for t in transactions
            if t.get("client_http_version", "")
        )
    )
    distinct_service_count = len(
        set(t.get("service", "") for t in transactions if t.get("service", ""))
    )

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

    distinct_file_count = len(
        set(file_hash for t in transactions for file_hash in t.get("file_hashes", []))
    )
    distinct_file_types = sorted(
        list(
            set(
                file_type.split(";")[0]
                for t in transactions
                for file_types in t.get("file_types", [])
                for file_type in file_types
                if file_type
            )
        )
    )
    distinct_file_type_count = len(distinct_file_types)
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

    key_domains = sorted(list(set(h for t in transactions for h in t.get("key_hostnames", []))))
    distinct_key_domain_count = len(key_domains)
    range_timestamp = int(transactions[-1]["timestamp"] - transactions[0]["timestamp"])
    time_taken_stats = get_numeric_stats(events=transactions, field="time_taken_ms")
    client_byte_stats = get_numeric_stats(events=transactions, field="client_bytes")
    server_byte_stats = get_numeric_stats(events=transactions, field="server_bytes")
    time_interval_sec_stats = get_numeric_stats( events=transactions[1:], field="time_interval_sec" )
    applications = list(set(t.get("application", "") for t in transactions if t.get("application", "")))
    unique_actions = set(t["action"] for t in transactions if "action" in t)

    if len(applications) > 1:
        raise Exception("[!] More than one application detected for the same useragent", key)

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
        summary.update({
            "domain": list(distinct_domains)[0] if distinct_domains else ""
        })

    summary.update(time_taken_stats)
    summary.update(client_byte_stats)
    summary.update(server_byte_stats)
    summary.update(time_interval_sec_stats)
    return summary


def aggregate_app_traffic(fields: List[str], input_path: str, output_path: str, min_transactions: int) -> None:
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
        summary = grab_application_summary(traffic_map=traffic_map, key=key, fields=fields)
        if summary["transactions"] > min_transactions:
            summaries.append(summary)
        else:
            # TODO: Format the applications that did not have enough traffic in a nicer way
            # logger.info(f"\t[x] Application = {summary['key']}
            # (Number of transactions {summary['transactions']} did not meet the minimum required amount)")
            pass

    save_json_data(summaries, output_path)
