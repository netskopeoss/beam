"""
Contains the code for parsing the .har files into Netskope Transaction logs format
"""

import json
import sys
from datetime import datetime
from typing import List
from beam.parser.models import NetskopeTransaction

def convert_list(entries: List) -> dict:
    """
    Convert a list of header entries to a dictionary.

    Args:
        entries (List): A list of header entries, each containing 'name' and 'value' keys.

    Returns:
        dict: A dictionary with header names as keys and their corresponding values.

    Raises:
        None
    """
    output = {}
    for entry in entries:
        output[entry["name"].lower()] = entry["value"]
    return output


def parse_har_log(file_path: str) -> List[NetskopeTransaction]:
    """
    Parse a .har file and convert it to a list of NetskopeTransaction objects.

    Args:
        file_path (str): The path to the .har file to parse.

    Returns:
        List[NetskopeTransaction]: A list of ZeekLog objects parsed from the .har file.

    Raises:
        None
    """
    entries = []
    with open(file=file_path, mode="r", encoding="utf-8-sig") as file:
        har_data = json.load(file)
        for entry in har_data["log"]["entries"]:
            http_request = entry["request"]
            http_response = entry["response"]
            req_headers = convert_list(http_request["headers"])
            resp_headers = convert_list(entry["response"].get("headers", []))
            data = {
                "timestamp": str(datetime
                    .strptime(entry["startedDateTime"]
                    .split(".")[0], "%Y-%m-%dT%H:%M:%S")),
                "day": "", #TODO: Figure out this later from the timestamp
                "hour": "",#TODO: Figure out this later from the timestamp
                "access_method": "Client",
                "useragent": req_headers.get("User-Agent", ""),
                "hostname": req_headers.get("Host",""),
                "referer": "",
                "uri_scheme": entry["request"]["url"].split(":")[0],
                "http_method": entry["request"]["method"],
                "http_status": str(entry["response"]["status"]),
                "rs_status": "",
                "ssl_ja3": "",
                "ssl_ja3s": "",
                "file_type": "",
                "traffic_type": "CloudApp",
                "client_http_version": entry["response"]["httpVersion"],
                "srcport": "", #TODO:
                "client_src_port": "",
                "client_dst_port": "",
                "client_connect_port": "",
                "server_src_port": "",
                "server_dst_port": "",
                "req_content_type": req_headers.get("Content-Type", ""),
                "resp_content_type": http_response["content"]["mimeType"],
                "server_ssl_error": "",
                "client_ssl_error": "",
                "error": "",
                "ssl_bypass": "",
                "ssl_bypass_reason": "",
                "ssl_fronting_error": "",
                "time_taken_ms": str(entry["time"]),
                "client_bytes": str(http_request["bodySize"]),
                "server_bytes": str(http_response["content"]["size"]),
                "file_sha256": "",
                "file_size": "",
                "url": entry["request"]["url"]
            }
            entries.append(
                NetskopeTransaction(**data)
            )
    return entries


if __name__ == "__main__":
    parsed_responses = parse_har_log(sys.argv[1])
    for response in parsed_responses:
        print(response.model_dump_json())
