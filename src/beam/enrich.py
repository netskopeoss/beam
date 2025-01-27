import logging
from os import environ
from typing import List
from beam import utils
from beam.mapper.mapper import query_user_agent_mapper

DATA_DIR = utils.get_project_root() / "data"
KEY_DOMAINS_FILE = DATA_DIR / "key_domains.json"
CLOUD_DOMAINS_FILE = DATA_DIR / "cloud_domains.json"

def load_cloud_app_domains(cloud_domains_file_path: str = CLOUD_DOMAINS_FILE) -> List[str]:
    """
    Helper function to load known cloud domains.

    Args:
        cloud_domains_file_path (str): The path to the JSON file containing cloud domains.

    Returns:
        List[str]: A list of cloud domain strings.

    Raises:
        None
    """
    cloud_domains = [d["domain"] for d in utils.load_json_file(cloud_domains_file_path)]
    return cloud_domains

def load_key_domains(key_domains_file_path: str = KEY_DOMAINS_FILE) -> List[str]:
    """
    Helper function to load key domains.

    Args:
        key_domains_file_path (str): The path to the JSON file containing key domains.

    Returns:
        List[str]: A list of key domain strings.

    Raises:
        None
    """
    key_domains = [d["domain"] for d in utils.load_json_file(key_domains_file_path)]
    return key_domains

def check_for_key_domains(domain: str) -> List[str]:
    """
    Checks to see if any of the relevant hostnames have been contacted.

    Args:
        domain (str): The domain to check against the key domains list.

    Returns:
        List[str]: A sorted list of key domains that match the given domain.

    Raises:
        None
    """
    if not domain:
        return []
    key_domains = load_key_domains(key_domains_file_path=KEY_DOMAINS_FILE)
    return sorted(list({h for h in key_domains if domain.endswith(h)}))

def get_traffic_type(domain: str) -> str:
    """
    Assign a cloud / non_cloud label based on the domain provider.

    Args:
        domain (str): The domain to check against the cloud domains list.

    Returns:
        str: 'cloud' if the domain is a known cloud domain, otherwise 'non_cloud'.

    Raises:
        None
    """
    if not domain:
        return None
    cloud_domains = load_cloud_app_domains()
    traffic_type = 'cloud' if any(d in domain for d in cloud_domains) else 'non_cloud'
    return traffic_type

def enrich_events(
        input_path: str,
        output_path: str,
        logger: logging.Logger
        ) -> None:
    """
    Take in a parsed Zeek JSON file and enrich it with application information.

    Args:
        input_path (str): The path to the input JSON file containing parsed Zeek events.
        output_path (str): The path to the output JSON file where enriched events will be saved.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        None

    Raises:
        None
    """
    events = utils.load_json_file(input_path)
    logger.info(f"Enriching file {input_path}")

    full_ua_list = [event['useragent'] for event in events]
    unique_ua_list = list(dict.fromkeys(full_ua_list).keys())
    mapper = query_user_agent_mapper(
        user_agents=unique_ua_list,
        llm_api_key=environ['GEMINI_API_KEY'],
        llm_selection="Gemini",
        logger=logger
    )
    mapper.save_results()
    application_map = mapper.hits

    hits = { mapping.user_agent_string: mapping.application.name for mapping in application_map }

    for event in events:
            if event['useragent'] in hits.keys():
                application = hits['useragent']
            else:
                application = "unknown"
                logger.info(f"User Agent from event was not resolved: {event['useragent']}")
            event.update({
                'application': application,
                'key_domains': check_for_key_domains(domain=event['domain']),
                'traffic_type': get_traffic_type(domain=event['domain'])
                })
    
    utils.save_json_data(events, output_path)


if __name__ == "__main__":
    pass