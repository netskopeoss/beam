import logging
from typing import List

from detector.utils import load_json_file
from mapper.data_sources import Mapping
from mapper.mapper import query_user_agent_mapper


def load_cloud_app_domains(cloud_domains_file_path: str) -> List[str]:
    """
    Helper function to load known cloud domains

    :return:
    """

    # TODO: Update this domain list with more cloud domains

    cloud_domains = [d["domain"] for d in load_json_file(cloud_domains_file_path)]
    return cloud_domains


def load_key_domains(key_domains_file_path) -> List[str]:
    """
    Helper function to load key domains

    :return:
    """

    # TODO: Update this domain list with more key domains

    key_domains = [d["domain"] for d in load_json_file(key_domains_file_path)]
    return key_domains


def check_for_key_domains(domain: str, key_domains_file_path) -> List[str]:
    """
    Checks to see if any of the relevant hostnames have been contacted

    :param domain:
    :param key_domains_file_path:
    :return:
    """
    if not domain:
        return []
    key_domains = load_key_domains(key_domains_file_path)
    return sorted(list({h for h in key_domains if domain.endswith(h)}))


def get_traffic_type(domain: str, cloud_domains_file_path) -> None | str:
    """
    Assign a cloud / non_cloud label based on the domain provider

    :param domain:
    :param cloud_domains_file_path:
    :return:
    """
    if not domain:
        return None
    else:
        cloud_domains = load_cloud_app_domains(cloud_domains_file_path)
        traffic_type = (
            "cloud" if any(d in domain for d in cloud_domains) else "non_cloud"
        )
        return traffic_type


def get_url_endpoint(domain, url):
    """

    :param domain:
    :param url:
    """
    new_url = "/".join(url.split("?")[0].split("/")[:4])
    return f"{domain}{new_url}"


def enrich_events(
    input_path: str,
    db_path,
    cloud_domains_file_path,
    key_domains_file_path,
    llm_api_key,
) -> dict:
    """
    Take in a parsed zeek json file and enrich it with application information.

    :param input_path:
    :param db_path:
    :param cloud_domains_file_path:
    :param key_domains_file_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Enriching file {input_path}")
    events = load_json_file(input_path)
    full_ua_list = [event["useragent"] for event in events]
    unique_ua_list = list(dict.fromkeys(full_ua_list).keys())

    mapper = query_user_agent_mapper(
        user_agents=unique_ua_list,
        db_path=db_path,
        llm_api_key=llm_api_key,
        llm_selection="Gemini",
        logger=logger,
    )

    hits = dict()
    for mapping in mapper.hits:
        if isinstance(mapping, dict):
            hits[mapping["user_agent"]] = mapping["application"]["name"]
        elif isinstance(mapping, Mapping):
            hits[mapping.user_agent_string] = mapping.application.name
        else:
            logger.error("[!!] Unknown data type returned...", type(mapping))
            exit()

    mapper.save_results()

    for event in events:
        url_endpoint = get_url_endpoint(event["domain"], event["url"])

        if event["useragent"] in hits.keys():
            application = hits[event["useragent"]]
        else:
            application = "unknown"
            logger.debug(
                f"User Agent from event was not resolved: {event['useragent']}"
            )

        event.update(
            {
                "application": application,
                # TODO: Add these once the mapper is working correctly
                # "version": (
                #     map_result["version"]
                #     if map_result and ("version" in map_result)
                #     else "unknown"
                # ),
                # "os": (
                #     map_result["os"]
                #     if map_result and ("os" in map_result)
                #     else "unknown"
                # ),
                # "vendor": (
                #     map_result["vendor"]
                #     if map_result and ("vendor" in map_result)
                #     else "unknown"
                # ),
                # "description": (
                #     map_result["description"]
                #     if map_result and ("description" in map_result)
                #     else ""
                # ),
                "key_hostnames": check_for_key_domains(
                    domain=event["domain"], key_domains_file_path=key_domains_file_path
                ),
                "traffic_type": get_traffic_type(
                    domain=event["domain"],
                    cloud_domains_file_path=cloud_domains_file_path,
                ),
                "url_endpoint": url_endpoint,
                "action": event["http_method"] + " " + url_endpoint,
            }
        )
    return events


if __name__ == "__main__":
    pass
