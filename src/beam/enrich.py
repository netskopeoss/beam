"""Enrich Module"""

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
import re
import string
from typing import List, Optional

from beam.detector.utils import load_json_file
from beam.mapper.mapper import query_user_agent_mapper


def load_cloud_app_domains(cloud_domains_file_path: str) -> List[str]:
    """
    Helper function to load known cloud domains.

    Args:
        cloud_domains_file_path (str): The path to the file containing cloud domains.

    Returns:
        List[str]: A list of cloud domains.
    """

    cloud_domains = [d["domain"] for d in load_json_file(cloud_domains_file_path)]
    return cloud_domains


def load_key_domains(key_domains_file_path) -> List[str]:
    """
    Helper function to load key domains.

    Args:
        key_domains_file_path (str): The path to the file containing key domains.

    Returns:
        List[str]: A list of key domains.
    """

    key_domains = [d["domain"] for d in load_json_file(key_domains_file_path)]
    return key_domains


def check_for_key_domains(domain: str, key_domains_file_path) -> List[str]:
    """
    Checks to see if any of the relevant hostnames have been contacted.

    Args:
        domain (str): The domain to check.
        key_domains_file_path (str): The path to the file containing key domains.

    Returns:
        List[str]: A list of key domains that match the given domain.
    """
    if not domain:
        return []
    key_domains = load_key_domains(key_domains_file_path)
    return sorted(list({h for h in key_domains if domain.endswith(h)}))


def get_traffic_type(domain: str, cloud_domains_file_path) -> None | str:
    """
    Assign a cloud / non_cloud label based on the domain provider.

    Args:
        domain (str): The domain to check.
        cloud_domains_file_path (str): The path to the file containing cloud domains.

    Returns:
        None | str: "cloud" if the domain is a known cloud domain, "non_cloud" otherwise, or None if the domain is empty.
    """
    if not domain:
        return None
    else:
        cloud_domains = load_cloud_app_domains(cloud_domains_file_path)
        traffic_type = (
            "cloud" if any(d in domain for d in cloud_domains) else "non_cloud"
        )
        return traffic_type


def get_url_endpoint(domain: str, url: str) -> str:
    """
    Construct the URL endpoint from the domain and URL.

    Args:
        domain (str): The domain of the URL.
        url (str): The full URL.

    Returns:
        str: The constructed URL endpoint.
    """
    new_url = "/".join(url.split("?")[0].split("/")[:4])
    return f"{domain}{new_url}"


def valid_useragent_string(useragent: str) -> Optional[str]:
    """
    Validate and filter out problematic user agent strings.

    Args:
        useragent (str): The user agent string to validate.

    Returns:
        Optional[str]: The validated user agent string, or None if it is considered problematic.
    """
    if not useragent:
        return None

    if re.match(r"^[\\S]*[\\*\\@\\$].*", useragent):
        return None

    if all(c in string.digits for c in useragent):
        return None

    if useragent in ("-", ".-", "-.", "-.-"):
        return None

    if useragent[0] == "{" and useragent[-1] == "}" and len(useragent) == 38:
        return None

    invalid_phrases = [
        "JPNET",
        "google-cloud-sdk gcloud",
        "gcloud/",
        "Visual Component",
        "ECAgent/",
        "ruxit/",
        "RuxitSynthetic",
        "NGL Client/",
        ";PID=",
        "Zimbra-ZCO",
    ]

    if any(p in useragent for p in invalid_phrases):
        return None

    return useragent


def enrich_events(
    input_path: str,
    db_path: str,
    cloud_domains_file_path: str,
    key_domains_file_path: str,
    llm_api_key: str,
    use_local_llm: bool = True,  # Default to local LLM
    remote_llm_type: str = None,
) -> dict:
    """
    Take in a parsed Zeek JSON file and enrich it with application information.

    Args:
        input_path (str): The path to the input JSON file.
        db_path (str): The path to the database file.
        cloud_domains_file_path (str): The path to the file containing cloud domains.
        key_domains_file_path (str): The path to the file containing key domains.
        llm_api_key (str): The API key for the language model.
        use_local_llm (bool): Whether to use local Llama model (default: True).
        remote_llm_type (str): Type of remote LLM to use if not using local (e.g., 'gemini').

    Returns:
        dict: The enriched events.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Enriching file {input_path}")
    events = load_json_file(input_path)
    full_ua_list = [
        event["useragent"]
        for event in events
        if valid_useragent_string(event["useragent"])
    ]
    unique_ua_list = list(dict.fromkeys(full_ua_list).keys())

    # Choose LLM selection based on configuration
    if use_local_llm:
        llm_selection = "Llama"
    elif remote_llm_type:
        llm_selection = remote_llm_type.capitalize()  # e.g., "gemini" -> "Gemini"
    else:
        llm_selection = "Llama"  # Default to local
    logger.info(f"Using LLM: {llm_selection} (use_local_llm={use_local_llm}, remote_llm_type={remote_llm_type})")
    
    hits, misses = query_user_agent_mapper(
        user_agents=unique_ua_list,
        db_path=db_path,
        llm_api_key=llm_api_key,
        llm_selection=llm_selection,
        logger=logger,
    )
    user_agents = [hit["user_agent_string"] for hit in hits]

    for event in events:
        if event["useragent"] in user_agents:
            hit = next(
                hit for hit in hits if hit["user_agent_string"] == event["useragent"]
            )
            event.update(
                {
                    "application": hit["application"],
                    "version": hit["version"],
                    "os": hit["operating_system"],
                    "vendor": hit["vendor"],
                    "description": hit["description"],
                }
            )
        else:
            event.update(
                {
                    "application": "unknown",
                    "version": "unknown",
                    "os": "unknown",
                    "vendor": "unknown",
                    "description": "unknown",
                }
            )
            logger.debug(
                f"User Agent from event was not resolved: {event['useragent']}"
            )
        uri = (event["uri"] if "uri" in event else event["url"]).split("?")[0]
        url_endpoint = get_url_endpoint(event["domain"], uri)
        event.update(
            {
                "key_hostnames": check_for_key_domains(
                    domain=event["domain"], key_domains_file_path=key_domains_file_path
                ),
                "traffic_type": get_traffic_type(
                    domain=event["domain"],
                    cloud_domains_file_path=cloud_domains_file_path,
                ),
                "url_endpoint": url_endpoint,
                "action": (event["http_method"] + " " + url_endpoint).strip(),
            }
        )
    return events
