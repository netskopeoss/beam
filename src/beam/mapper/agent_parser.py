import httpagentparser
import time
from ua_parser import user_agent_parser
from user_agents import parse
from typing import List, Dict
from beam.mapper.data_sources import APIDataSource


def map_item(user_agent: str) -> Dict:
    """
    Map a user agent string and return a dictionary with
    application, version, vendor, os, and description.

    Args:
        user_agent (str): The user agent string to parse.

    Returns:
        Dict: A dictionary containing information about the user agent, including
        application name, version, vendor, description, and OS name.

    Raises:
        None
    """

    # Setting the defaults if they are not found
    application = 'unknown'
    version = 'unknown'
    os = 'unknown'
    vendor = 'unknown'
    description = 'unknown'

    if not user_agent:
        return {
            "user_agent": user_agent,
            "application": {
                'name': application,
                'version': version,
                'vendor': vendor,
                'description': description
            },
            "os": {
                'name': os
            }
        }

    application = str(parse(user_agent).browser.family).strip()
    version = str(parse(user_agent).browser.version_string).strip()
    os = str(parse(user_agent).os.family).strip() + ' ' + str(parse(user_agent).os.version_string).strip()
    vendor = str(parse(user_agent).device.brand).strip()

    if application != 'Other':
        return {
            "user_agent": user_agent,
            "application": {
                'name': application,
                'version': version,
                'vendor': vendor,
                'description': description
            },
            "os": {
                'name': os
            }
        }

    parsed = user_agent_parser.Parse(user_agent)
    application = parsed['user_agent']['family'].strip()
    major = parsed['user_agent'].get('major', '')
    minor = parsed['user_agent'].get('minor', '')
    patch = parsed['user_agent'].get('patch', '')
    version_parts = [major, minor, patch]
    version = '.'.join(filter(None, version_parts)).strip()

    os_family = parsed['os']['family'].strip()
    major = parsed['os'].get('major', '')
    minor = parsed['os'].get('minor', '')
    patch = parsed['os'].get('patch', '')
    version_parts = [major, minor, patch]
    os = os_family + ' ' + '.'.join(filter(None, version_parts)).strip()

    if application != 'Other':
        return {
            "user_agent": user_agent,
            "application": {
                'name': application,
                'version': version,
                'vendor': vendor,
                'description': description
            },
            "os": {
                'name': os
            }
        }

    application = httpagentparser.simple_detect(user_agent)[1].strip()
    if '.' in application.split(' ')[-1]:
        application = ' '.join(application.split(' ')[:-1])
        version = application.split(' ')[-1]
    os = httpagentparser.simple_detect(user_agent)[0].strip()

    result = {
        "user_agent": user_agent,
        "application": {
            'name': application,
            'version': version,
            'vendor': vendor,
            'description': description
        },
        "os": {
            'name': os
        }
    }

    return result


class AgentParser(APIDataSource):
    """Class to use external python libraries to gather application information
    from user agents.
    """
    def search(self) -> None:
        """
        Parse each user agent in self.query_input using map_item. Populate self.hits
        with the results and self.misses with user agents that failed to parse.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        query_start = time.perf_counter()
        for user_agent in self.query_input:
            parser_result = map_item(user_agent)
            if parser_result:
                self.hits.append(parser_result)
            else:
                self.misses.append(user_agent)
        query_stop = time.perf_counter()
        self.query_time = query_stop - query_start


def query_agent_parser(user_agents:List[str]) -> AgentParser:
    """
    Create an AgentParser instance with the given user agent strings, run its
    search method, and return the populated AgentParser object.

    Args:
        user_agents (List[str]): A list of user agent strings to parse.

    Returns:
        AgentParser: The AgentParser object containing hits for successfully parsed
        user agents and misses for those that failed to parse.

    Raises:
        None
    """
    api_parser = AgentParser(query_input=user_agents)
    api_parser.search()
    return api_parser
