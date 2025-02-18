"""Zeek module"""

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
import logging
import subprocess
from pathlib import Path
from typing import Dict, List

# TODO: Future improvements
#  - Handle encrypted packet captures via this method as well
#  https://docs.zeek.org/en/master/frameworks/tls-decryption.html
#  - Also consider suggesting a method to decrypt ALL the traffic from a
#  machine, like a VM, to show how to capture traffic before feeding
#  it to this system
#  - Add JA4 fingerprint


def run_zeek(file_path: str) -> str:
    """
    Wrapper to run the zeek command against a file path.

    Args:
        file_path (str): The path to the PCAP file to process.

    Returns:
        str: The path to the directory where Zeek output is saved.

    Raises:
        None
    """
    logger = logging.getLogger(__name__)
    output_path = file_path.replace(".pcap", "").replace("input", "zeek")
    logger.info(
        f"[X] Running {file_path} through zeek and outputting the results here: {output_path}"
    )
    Path(output_path).mkdir(parents=True, exist_ok=True)
    command_to_run = [
        "zeek",
        "-C",
        "-r",
        file_path,
        "Log::default_logdir=" + output_path,
        "LogAscii::use_json=T",
        "local",
    ]
    logger.info(f"[x] Running {' '.join(command_to_run)}")
    subprocess.run(command_to_run)
    return output_path


def grab_zeek_log(dir_path: str, log_name: str) -> Dict:
    """
    Helper function to parse through a zeek log and return a python dictionary.

    Args:
        dir_path (str): The directory path where the Zeek log is located.
        log_name (str): The name of the Zeek log file to parse.

    Returns:
        dict: A dictionary containing parsed log data, keyed by 'uid'.

    Raises:
        None
    """

    logger = logging.getLogger(__name__)
    log_path = f"{dir_path}/{log_name}"
    log_data = dict()

    try:
        with open(log_path) as _file:
            for line in _file.readlines():
                e = json.loads(line)
                if e["uid"] in log_data:
                    log_data[e["uid"]].append(e)
                else:
                    log_data[e["uid"]] = [e]

    except Exception as e:
        logger.info(f"[x] Error opening {log_path}")

    return log_data


def process_zeek_output(input_path: str) -> List:
    """
    Parse the Zeek output logs and save the parsed results to a JSON file.

    Args:
        input_path (str): The path to the directory containing Zeek output logs.

    Returns:
        None

    Raises:
        None
    """
    logger = logging.getLogger(__name__)
    http_data = grab_zeek_log(input_path, log_name="http.log")
    conn_data = grab_zeek_log(input_path, log_name="conn.log")
    files_data = grab_zeek_log(input_path, log_name="files.log")
    ssl_data = grab_zeek_log(input_path, log_name="ssl.log")

    parsed_result = []

    for _uid in http_data.items():
        try:
            http_row = http_data[_uid][0]
            connection = conn_data.get(_uid, None)[0]
            files = files_data.get(_uid, None)

            if "host" in http_row:
                domain = http_row["host"].split(":")[0]
            else:
                domain = ""

            http_status = (
                http_row.get("status_code", None)
                if http_row.get("status_code", None)
                else None
            )
            status_msg = (
                http_row.get("status_msg", None)
                if http_row.get("status_msg", None)
                else None
            )
            req_types = http_row.get("orig_mime_types", [])
            resp_types = http_row.get("resp_mime_types", [])

            # TODO: We are only grabbing one entry here - need to identify why the discrepancy

            req_content_type = req_types[0] if len(req_types) > 0 else None
            resp_content_type = resp_types[0] if len(resp_types) > 0 else None

            # TODO: these do not seem like files - need to filter them better if NSKP output is actual files

            file_types = (
                sorted(
                    list({c.get("mime_type", "") for c in files} if files else set())
                ),
            )
            file_sizes = (
                sorted(
                    list({c.get("total_bytes", 0) for c in files} if files else set())
                ),
            )

            # TODO: We are only grabbing one entry here - need to identify why the discrepancy

            file_type = file_types[0] if len(file_types) > 0 else None
            file_size = file_sizes[0] if len(file_sizes) > 0 else None

            # TODO: At the moment, we are only dealing with http traffic
            #     Need to identify cases where this might be https
            uri_scheme = "http"
            referrer = http_row.get("referrer", "")
            is_referred = True if referrer else False

            useragent = http_row.get("user_agent", "")

            parsed_log_line = {
                "timestamp": connection.get("ts", 0),
                "useragent": useragent,
                "domain": domain,
                "referrer": referrer,
                "is_referred": is_referred,
                "uri": http_row.get("uri", ""),
                "http_method": http_row.get("method", ""),
                # 'client_http_version': http_row.get('version', ''),
                "src_ip": http_row.get("id.orig_h", ""),
                "src_port": http_row.get("id.orig_p", ""),
                "dst_ip": http_row.get("id.resp_h", ""),
                "dst_port": http_row.get("id.resp_p", ""),
                "client_bytes": connection["orig_bytes"],
                "server_bytes": connection["resp_bytes"],
                "time_taken_ms": connection["duration"] * 1000.0,
                "service": connection.get("service", ""),
                "http_status": http_status,
                "status_msg": status_msg,
                "req_types": req_types,
                "resp_types": resp_types,
                "req_content_type": req_content_type,
                "resp_content_type": resp_content_type,
                "file_types": file_types,
                "file_sizes": file_sizes,
                "file_hashes": sorted(
                    list({c.get("md5", "") for c in files} if files else set())
                ),
                "file_type": file_type,
                "file_size": file_size,
                # 'uri_scheme': uri_scheme
            }

            parsed_result.append(parsed_log_line)

        except Exception as e:
            logger.error(f"[!!] Exception: {e}")
    return parsed_result
