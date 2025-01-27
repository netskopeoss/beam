import statistics
import logging
from beam import utils
from urllib.parse import urlparse


# TODO: Set a global config to parse JA3 / JA3s values if they have provided them

# F.collect_set('ssl_ja3').alias('ja3_values'),
# F.collect_set('ssl_ja3s').alias('ja3s_values'),
# F.countDistinct('ssl_ja3').alias('ja3_cnt'),
# F.countDistinct('ssl_ja3s').alias('ja3s_cnt'),

def get_numeric_stats(events, field):
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

    return {
        f'avg_{field}': statistics.mean(values) if len(values) > 0 else 0,
        f'std_{field}': statistics.stdev(values) if len(values) > 1 else 0,
        f'median_{field}': statistics.median(values) if len(values) > 0 else 0,
        f'range_{field}': max(values) - min(values) if len(values) > 0 else 0,
        f'max_{field}': max(values) if len(values) > 0 else 0,
        f'min_{field}': min(values) if len(values) > 0 else 0,
        f'sum_{field}': sum(values) if len(values) > 0 else 0,
    }


def log_inventory():
    """
    Placeholder function to support the ability to inventory applications running and their behavior.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    pass


def grab_application_summary(traffic_map, key):
    """
    Generate a summary of application traffic based on the provided traffic map and key.

    Args:
        traffic_map (dict): A dictionary mapping application keys to lists of transaction events.
        key (str): The key identifying the application in the traffic map.

    Returns:
        dict: A dictionary containing a summary of the application's traffic, including metrics
              such as transaction count, distinct domain count, HTTP methods, status codes, and more.

    Raises:
        None
    """
    transactions = sorted(traffic_map[key], key=lambda x: x['timestamp'])
    for i, transaction in enumerate(transactions):
        if i >= 1:
            transaction['time_interval_sec'] = transaction['timestamp'] - transactions[i - 1]['timestamp']
        else:
            transaction['time_interval_sec'] = 0.0

    transaction_count = len(transactions)

    distinct_domain_count = len(set(t.get('domain', '') for t in transactions if t.get('domain', '')))
    distinct_uri_count = len(set(t.get('uri', '') for t in transactions if t.get('domain', '')))
    distinct_referrer_count = len(set(urlparse(t.get('referrer', '')).netloc for t in transactions if t.get('referrer', '')))
    distinct_methods = sorted(list(set(t.get('method', '') for t in transactions if t.get('method', ''))))
    distinct_method_count = len(distinct_methods)
    distinct_statuses = sorted(list(set(t.get('status_code', '') for t in transactions if t.get('status_code', ''))))
    distinct_status_count = len(distinct_statuses)
    distinct_version_count = len(set(t.get('http_version', '') for t in transactions if t.get('http_version', '')))
    distinct_service_count = len(set(t.get('service', '') for t in transactions if t.get('service', '')))

    distinct_req_types = sorted(list(set(req_type.split(';')[0] for t in transactions for req_type in t.get('req_types', []))))
    distinct_req_type_count = len(distinct_req_types)

    distinct_resp_types = sorted(list(set(resp_type.split(';')[0] for t in transactions for resp_type in t.get('resp_types', []))))
    distinct_resp_type_count = len(distinct_resp_types)

    distinct_file_count = len(set(file_hash for t in transactions for file_hash in t.get('file_hashes', [])))
    distinct_file_types = sorted(list(set(file_type.split(';')[0] for t in transactions for file_type in t.get('file_types', []) if file_type)))
    distinct_file_type_count = len(distinct_file_types)

    referred_traffic_percent = (len([1 for t in transactions if 'referrer' in t and t['referrer']]) * 100.0) / transaction_count
    cloud_traffic_percent = (len([1 for t in transactions if 'traffic_type' in t and t['traffic_type'] == 'cloud']) * 100.0) / transaction_count
    web_traffic_percent = (len([1 for t in transactions if 'traffic_type' in t and t['traffic_type'] == 'non_cloud']) * 100.0) / transaction_count

    key_domains = sorted(list(set(h for t in transactions for h in t.get('key_domains', []))))
    distinct_key_domain_count = len(key_domains)

    range_timestamp = int(transactions[-1]['timestamp'] - transactions[0]['timestamp'])

    time_taken_stats = get_numeric_stats(events=transactions, field='time_taken_ms')
    client_byte_stats = get_numeric_stats(events=transactions, field='client_bytes')
    server_byte_stats = get_numeric_stats(events=transactions, field='server_bytes')
    time_interval_sec_stats = get_numeric_stats(events=transactions[1:], field='time_interval_sec')

    summary = {
        'key': key,
        'transactions': transaction_count,
        'domain_cnt': distinct_domain_count,
        'referer_domain_cnt': distinct_referrer_count,
        'distinct_key_domain_count': distinct_key_domain_count,
        'http_method_cnt': distinct_method_count,
        'http_status_cnt': distinct_status_count,
        'req_content_type_cnt': distinct_req_type_count,
        'resp_content_type_cnt': distinct_resp_type_count,
        'http_methods': distinct_methods,
        'http_statuses': distinct_statuses,
        'req_content_types': distinct_req_types,
        'resp_content_types': distinct_resp_types,
        'cloud_traffic_pct': cloud_traffic_percent,
        'web_traffic_pct': web_traffic_percent,
        'refered_traffic_pct': referred_traffic_percent,
        'key_hostnames': key_domains,
        'key_hostname_cnt': distinct_key_domain_count,
        'range_timestamp': range_timestamp,

        # TODO: Add this back in to both the places
        # 'distinct_uri_count': distinct_uri_count,

        # 'distinct_version_count': distinct_version_count,
        # 'distinct_file_count': distinct_file_count,
        # 'distinct_file_types': distinct_file_types,
        # 'distinct_file_type_count': distinct_file_type_count,
        # 'distinct_service_count': distinct_service_count,
    }

    summary.update(time_taken_stats)
    summary.update(client_byte_stats)
    summary.update(server_byte_stats)
    summary.update(time_interval_sec_stats)
    return summary


def aggregate_app_traffic(app_field, input_path, output_path):
    """
    Aggregate application traffic data from enriched events and save the summaries to a JSON file.

    Args:
        app_field (str): The field in the events to use as the application identifier.
        input_path (str): The path to the input JSON file containing enriched events.
        output_path (str): The path to the output JSON file where summaries will be saved.

    Returns:
        None

    Raises:
        None
    """
    logger = logging.getLogger(__name__)
    traffic_map = dict()

    enriched_events = utils.load_json_file(input_path)
    summaries = []

    # TODO: Have a config to handle invalid summaries
    invalid_summaries = []

    for e in enriched_events:
        if app_field in e:
            # TODO: Handle cases where the app_field / user agent is empty - take this in as a config

            key = f"'{e[app_field]}' on '{e['src_ip']}'"

            if key in traffic_map:
                traffic_map[key].append(e)
            else:
                traffic_map[key] = [e]

    logger.info(f"[x] Found a total of {len(traffic_map)} distinct values for '{app_field}' in provided event file")

    for key in traffic_map:
        summary = grab_application_summary(traffic_map=traffic_map, key=key)

        # TODO: Set this as a global config
        if summary['transactions'] > 100:
            summaries.append(summary)
        else:
            # TODO: Format the applications that did not have enough traffic in a nicer way
            # print(f"\t[x] Application = {summary['key']}
            # (Number of transactions {summary['transactions']} did not meet the minimum required amount)")
            pass

    utils.save_json_data(summaries, output_path)
