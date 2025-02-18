"""Har parsing module"""

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

import json
import sys
from datetime import datetime
from typing import Dict, List
from beam.parser.models import Transaction

def convert_list(entries: List) -> Dict:
    """
    Convert a list of header entries to a dictionary.

    Args:
        entries (List): A list of header entries, each containing 'name' and 'value' keys.

    Returns:
        output: A dictionary with header names as keys and their corresponding values.

    Raises:
        None
    """
    output = {}
    for entry in entries:
        output[entry["name"].lower()] = entry["value"]
    return output


def parse_har_log(file_path: str) -> List[Transaction]:
    """
    Parse a .har file and convert it to a list of NetskopeTransaction objects.

    Args:
        file_path (str): The path to the .har file to parse.

    Returns:
        List[Transaction]: A list of ZeekLog objects parsed from the .har file.

    Raises:
        None
    """
    entries = []
    statuses_to_avoid = [
        "999"  # Error
    ]
    methods_to_avoid = [
        "CONNECT"
    ]
    with open(file=file_path, mode="r", encoding="utf-8-sig") as file:
        har_data = json.load(file)
        for entry in har_data["log"]["entries"]:
            if ("request" in entry) and ("response" in entry):
                http_request = entry["request"]
                http_response = entry["response"]

                http_status = str(http_response["status"])
                http_method = http_request["method"]

                if (http_status not in statuses_to_avoid) and (http_method not in methods_to_avoid):
                    if ("headersSize" in http_request) and http_request["headersSize"] and (http_request["headersSize"] >= 0):
                        req_header_size = http_request["headersSize"]
                    else:
                        req_header_size = len('\n'.join([e['name'] + ' ' + e['value'] for e in entry['request']['headers']]))

                    if ("headersSize" in http_response) and http_response["headersSize"] and (http_response["headersSize"] >= 0):
                        resp_header_size = http_response["headersSize"]
                    else:
                        resp_header_size = len('\n'.join([e['name'] + ' ' + e['value'] for e in entry['response']['headers']]))

                    req_headers = convert_list(http_request.get("headers", []))
                    resp_headers = convert_list(http_response.get("headers", []))
                    timestamp = datetime.strptime(
                        entry["startedDateTime"].split(".")[0].split("+")[0],
                        "%Y-%m-%dT%H:%M:%S"
                    ).timestamp()
                    referer = req_headers.get('referer', '')
                    url = http_request.get("url", "")
                    hostname = req_headers.get("host", "").split(":")[0]
                    try:
                        uri = url.split(hostname)[1].split('?')[0]
                    except Exception:
                        uri = ''
                    data = {
                        "timestamp": timestamp,
                        "useragent": req_headers.get("user-agent", ""),
                        "hostname": hostname,
                        "domain": hostname,
                        "uri_scheme": http_request["url"].split(":")[0],
                        "http_method": http_method,
                        "http_status": http_status,
                        "client_http_version": http_response["httpVersion"],
                        "req_content_type": req_headers.get("content-type", "").split(';')[0],
                        "resp_content_type": http_response["content"]["mimeType"].split(';')[0],
                        "time_taken_ms":  int(entry["time"]),
                        # Confirmed with Louis Wu that it is the sum of both the body and headers
                        "client_bytes": float(http_request["bodySize"] + req_header_size),
                        "server_bytes": float(http_response["bodySize"] + resp_header_size),
                        "referer": referer,
                        "referer_domain": referer.split('/')[2].strip() if len(referer.split('/')) >= 3 else '',
                        "url": url,
                        "uri": uri,
                        "src_ip": entry['_clientAddress'] if '_clientAddress' in entry else 'Unknown'
                    }
                    entries.append(
                        Transaction(**data)
                    )
    return entries


if __name__ == "__main__":
    parsed_responses = parse_har_log(sys.argv[1])
    for response in parsed_responses:
        print(response.model_dump_json())
