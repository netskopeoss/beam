import json
import sys
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel


class ZeekLog(BaseModel):
    timestamp: float
    useragent: str
    domain: str
    referrer: str
    url: str
    http_method: str
    http_version: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    client_bytes: int
    server_bytes: int
    time_taken_ms: float
    service: Optional[str]
    status_code: int
    status_msg: str
    req_types: List[str]
    resp_types: List[str]
    file_types: List[str]
    file_sizes: List[int]
    file_hashes: List[str]


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


def parse_har_log(file_path: str) -> List[ZeekLog]:
    """
    Parse a .har file and convert it to a list of ZeekLog objects.

    Args:
        file_path (str): The path to the .har file to parse.

    Returns:
        List[ZeekLog]: A list of ZeekLog objects parsed from the .har file.

    Raises:
        None
    """
    with open(file=file_path, mode="r", encoding="utf-8-sig") as file:
        har_data = json.load(file)
        entries = []
        for entry in har_data["log"]["entries"]:
            connection = entry["timings"]
            http_request = entry["request"]
            # Set this to None, please revisit this in the future to see if it can be improved
            files = None
            status_code = entry["response"].get("status", 0)
            status_msg = entry["response"].get("statusText", "")
            req_headers = convert_list(http_request["headers"])
            resp_headers = convert_list(entry["response"].get("headers", []))

            req_types = [req_headers.get("content-type", "").split(";")[0]]
            resp_types = [resp_headers.get("Content-Type", "").split(";")[0]]

            epoch_time = (
                datetime.strptime(
                    entry["startedDateTime"].split(".")[0], "%Y-%m-%dT%H:%M:%S"
                )
                - datetime(1970, 1, 1)
            ).total_seconds()

            useragent = req_headers.get("user-agent", "")
            if (
                useragent
                == "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko"
            ):
                useragent = "PyCharm 1.45"

            entries.append(
                ZeekLog(
                    timestamp=epoch_time,
                    useragent=useragent,
                    domain=req_headers.get("host", ""),
                    referrer=req_headers.get("referer", ""),
                    url=urlparse(http_request.get("url", "")).path,
                    http_method=http_request.get("method", ""),
                    http_version=http_request.get("httpVersion", ""),
                    src_ip=entry.get("_clientAddress", ""),
                    src_port=entry.get("_clientPort", 0),
                    dst_ip=entry.get("_serverAddress", ""),
                    dst_port=entry.get("_serverPort", 0),
                    client_bytes=http_request.get("bodySize", 0),
                    server_bytes=entry["response"].get("bodySize", 0),
                    time_taken_ms=entry.get("time", 0),
                    service=entry.get("_clientName", ""),
                    status_code=status_code,
                    status_msg=status_msg,
                    req_types=req_types,
                    resp_types=resp_types,
                    file_types=sorted(list({files} if files else set())),
                    file_sizes=sorted(
                        list(
                            {entry["response"].get("content", {}).get("size", 0)}
                            if files
                            else set()
                        )
                    ),
                    file_hashes=sorted(
                        list(
                            {entry["response"].get("content", {}).get("text", "")}
                            if files
                            else set()
                        )
                    ),
                )
            )
        return entries


if __name__ == "__main__":
    parsed_responses = parse_har_log(sys.argv[1])
    for response in parsed_responses:
        print(response.model_dump_json())
