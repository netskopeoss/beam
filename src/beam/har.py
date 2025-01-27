import json
from typing import List, Optional
from pydantic import BaseModel
from urllib.parse import urlparse

class ZeekLog(BaseModel):
    timestamp: str
    useragent: str
    domain: str
    referrer: str
    uri: str
    method: str
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
    req_types: str
    resp_types: str
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
    with open(file=file_path, mode='r') as file:
        har_data = json.load(file)
        entries = []
        for entry in har_data['log']['entries']:
            connection = entry['timings']
            httpRequest = entry['request']
            # Set this to None, please revisit this in the future to see if it can be improved
            files = None
            status_code = entry['response'].get('status', 0)
            status_msg = entry['response'].get('statusText', '')
            reqHeaders = convert_list(httpRequest["headers"])
            respHeaders = convert_list(entry['response'].get('headers', []))

            req_types = reqHeaders.get('content-type', '')
            resp_types = respHeaders.get('Content-Type', '')
            
            entries.append(
                ZeekLog(
                    timestamp=entry['startedDateTime'],
                    useragent=reqHeaders.get('user-agent', ''),
                    domain=reqHeaders.get('host', ''),
                    referrer=reqHeaders.get('referer', ''),
                    uri=urlparse(httpRequest.get('url', '')).path,
                    method=httpRequest.get('method', ''),
                    http_version=httpRequest.get('httpVersion', ''),
                    src_ip=entry.get('_clientAddress', ''),
                    src_port=entry.get('_clientPort', 0),
                    dst_ip=entry.get('_serverAddress', ''),
                    dst_port=entry.get('_serverPort', 0),
                    client_bytes=httpRequest.get('bodySize', 0),
                    server_bytes=entry['response'].get('bodySize', 0),
                    time_taken_ms=entry.get('time', 0),
                    service=entry.get('_clientName', ''),
                    status_code=status_code,
                    status_msg=status_msg,
                    req_types=req_types,
                    resp_types=resp_types,
                    file_types=sorted(list({files} if files else set())),
                    file_sizes=sorted(list({entry['response'].get('content', {}).get('size', 0)} if files else set())),
                    file_hashes=sorted(list({entry['response'].get('content', {}).get('text', '')} if files else set()))
                )
            )
        return entries

if __name__ == "__main__":
    import sys
    parsed_responses = parse_har_log(sys.argv[1])
    for response in parsed_responses:
        print(response.model_dump_json())
