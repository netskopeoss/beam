"""Agent Parser Module"""

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

import time
from typing import List

import httpagentparser
from ua_parser import user_agent_parser
from user_agents import parse

from beam.mapper.data_sources import APIDataSource, Application, Mapping, OperatingSystem


def map_item(user_agent: str) -> Mapping | None:
    """
    Map a user agent string and return a Mapping object with
    application, version, vendor, os, and description.

    Args:
        user_agent (str): The user agent string to parse.

    Returns:
        Mapping: An object containing information about the user agent, including
        application name, version, vendor, description, and OS name.

    Raises:
        None
    """

    # Setting the defaults if they are not found
    application = "unknown"
    version = "unknown"
    os = "unknown"
    vendor = "unknown"
    description = "unknown"

    if not user_agent:
        return None

    application = str(parse(user_agent).browser.family).strip()
    version = str(parse(user_agent).browser.version_string).strip()
    os = (
        str(parse(user_agent).os.family).strip()
        + " "
        + str(parse(user_agent).os.version_string).strip()
    )
    vendor = str(parse(user_agent).device.brand).strip()

    if application not in ("Other", "unknown", "Unknown Browser"):
        app = Application(
            name=application, version=version, vendor=vendor, description=description
        )
        operating_system = OperatingSystem(name=os)
        return Mapping(
            user_agent_string=user_agent,
            application=app,
            operatingsystem=operating_system,
        )

    parsed = user_agent_parser.Parse(user_agent)
    application = parsed["user_agent"]["family"].strip()
    major = parsed["user_agent"].get("major", "")
    minor = parsed["user_agent"].get("minor", "")
    patch = parsed["user_agent"].get("patch", "")
    version_parts = [major, minor, patch]
    version = ".".join(filter(None, version_parts)).strip()

    os_family = parsed["os"]["family"].strip()
    major = parsed["os"].get("major", "")
    minor = parsed["os"].get("minor", "")
    patch = parsed["os"].get("patch", "")
    version_parts = [major, minor, patch]
    os = os_family + " " + ".".join(filter(None, version_parts)).strip()

    if application not in ("Other", "unknown", "Unknown Browser"):
        app = Application(
            name=application, version=version, vendor=vendor, description=description
        )
        operating_system = OperatingSystem(name=os)
        return Mapping(
            user_agent_string=user_agent,
            application=app,
            operatingsystem=operating_system,
        )

    application = httpagentparser.simple_detect(user_agent)[1].strip()
    if "." in application.split(" ")[-1]:
        application = " ".join(application.split(" ")[:-1])
        version = application.split(" ")[-1]
    os = httpagentparser.simple_detect(user_agent)[0].strip()

    if application not in ("Other", "unknown", "Unknown Browser"):
        app = Application(
            name=application, version=version, vendor=vendor, description=description
        )
        operating_system = OperatingSystem(name=os)
        return Mapping(
            user_agent_string=user_agent,
            application=app,
            operatingsystem=operating_system,
        )
    else:
        return None


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


def query_agent_parser(user_agents: List[str]) -> AgentParser:
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
