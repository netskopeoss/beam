import logging
import re
import string
from typing import List
from detector.utils import load_json_file
from mapper.mapper import query_user_agent_mapper


def load_cloud_app_domains(cloud_domains_file_path: str) -> List[str]:
    """
    Helper function to load known cloud domains

    :return:
    """
    cloud_domains = [d["domain"] for d in load_json_file(cloud_domains_file_path)]
    return cloud_domains


def load_key_domains(key_domains_file_path) -> List[str]:
    """
    Helper function to load key domains

    :return:
    """
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


def valid_useragent_string(useragent):
    """
    Remove a few problematic user agent strings

    """
    if not useragent:
        return None

    if re.match(r"^[\\S]*[\\*\\@\\$].*", useragent):
        return None

    if all(c in string.digits for c in useragent):
        return None

    if useragent in ('-', '.-', '-.', '-.-'):
        return None

    if useragent[0] == '{' and useragent[-1] == '}' and len(useragent) == 38:
        return None

    invalid_phrases = [
        'JPNET', 'google-cloud-sdk gcloud', 'gcloud/', 'Visual Component',
        'ECAgent/', 'ruxit/', 'RuxitSynthetic', 'NGL Client/', ';PID=', 'Zimbra-ZCO'
    ]

    if any(p in useragent for p in invalid_phrases):
        return None

    return useragent


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
    full_ua_list = [event["useragent"] for event in events if valid_useragent_string(event["useragent"])]
    unique_ua_list = list(dict.fromkeys(full_ua_list).keys())

    hits, misses = query_user_agent_mapper(
        user_agents=unique_ua_list,
        db_path=db_path,
        llm_api_key=llm_api_key,
        llm_selection="Gemini",
        logger=logger,
    )
    user_agents = [hit["user_agent_string"] for hit in hits]

    for event in events:
        if event["useragent"] in user_agents:
            hit = next(
                hit for hit in hits if hit["user_agent_string"] == event["useragent"]
            )
            event.update({
                    "application": hit["application"],
                    "version": hit["version"],
                    "os": hit["operating_system"],
                    "vendor": hit["vendor"],
                    "description": hit["description"]
                })
        else:
            event.update({
                "application": "unknown",
                "version": "unknown",
                "os": "unknown",
                "vendor": "unknown",
                "description": "unknown"
            })
            logger.debug(
                f"User Agent from event was not resolved: {event['useragent']}"
            )
        uri = (event["uri"] if "uri" in event else event["url"]).split('?')[0]
        url_endpoint = get_url_endpoint(event["domain"], uri)
        event.update(
            {
                "key_hostnames": check_for_key_domains(
                    domain=event["domain"],
                    key_domains_file_path=key_domains_file_path
                ),
                "traffic_type": get_traffic_type(
                    domain=event["domain"],
                    cloud_domains_file_path=cloud_domains_file_path
                ),
                "url_endpoint": url_endpoint,
                "action": (event["http_method"] + " " + url_endpoint).strip(),
            }
        )
    return events


if __name__ == "__main__":
    pass
