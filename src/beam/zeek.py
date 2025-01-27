import subprocess
import json
import logging
from pathlib import Path
from beam.utils import save_json_data

# TODO: Future improvments
#  - Handle encrypted packet captures via this method as well
#  https://docs.zeek.org/en/master/frameworks/tls-decryption.html
#  - Also consider suggesting a method to decrypt ALL the traffic from a
#  machine, like a VM, to show how to capture traffic before feeding
#  it to this system
#  - Add JA4 fingerprint

def run_zeek(
        file_path: str,
        logger: logging.Logger
        ) -> str:
    """
    Wrapper to run the zeek command against a file path.

    Args:
        file_path (str): The path to the PCAP file to process.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        str: The path to the directory where Zeek output is saved.

    Raises:
        None
    """
    output_path = file_path.replace('.pcap', '').replace('input_pcaps', 'zeek')
    logger.info(f"[X] Running {file_path} through zeek and outputting the results here: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    command_to_run = [
        'zeek', '-C', '-r', file_path, 'Log::default_logdir=' + output_path, 'LogAscii::use_json=T', 'local'
    ]
    logger.info(f"[x] Running {' '.join(command_to_run)}")
    subprocess.run(command_to_run)
    return output_path

def grab_zeek_log(
        dir_path: str,
        log_name: str,
        logger: logging.Logger
        ) -> dict:
    """
    Helper function to parse through a zeek log and return a python dictionary.

    Args:
        dir_path (str): The directory path where the Zeek log is located.
        log_name (str): The name of the Zeek log file to parse.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        dict: A dictionary containing parsed log data, keyed by 'uid'.

    Raises:
        None
    """
    log_path = f"{dir_path}/{log_name}"
    log_data = dict()

    logger.info(f"[x] Processing {log_name} from {dir_path}")
    try:
        with open(log_path) as _file:
            for line in _file.readlines():
                e = json.loads(line)
                if e['uid'] in log_data:
                    log_data[e['uid']].append(e)
                else:
                    log_data[e['uid']] = [e]

    except Exception as e:
        logger.error(f"[x] Error opening {log_path}", str(e))

    return log_data

def process_zeek_output(
        input_path: str,
        output_path: str,
        logger: logging.Logger
        ) -> None:
    """
    Parse the Zeek output logs and save the parsed results to a JSON file.

    Args:
        input_path (str): The path to the directory containing Zeek output logs.
        output_path (str): The path to the output JSON file where parsed results will be saved.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        None

    Raises:
        None
    """
    http_data = grab_zeek_log(
        input_path,
        log_name="http.log",
        logger=logger
        )
    
    conn_data = grab_zeek_log(
        input_path,
        log_name="conn.log",
        logger=logger
        )
    
    files_data = grab_zeek_log(
        input_path,
        log_name="files.log",
        logger=logger
        )
    
    ssl_data = grab_zeek_log(
        input_path,
        log_name="ssl.log",
        logger=logger
        )
    
    parsed_result = []

    for _uid in http_data:
        try:
            http_row = http_data[_uid][0]
            connection = conn_data.get(_uid, None)[0]
            files = files_data.get(_uid, None)

            if 'host' in http_row:
                domain = http_row['host'].split(':')[0]
            else:
                domain = ''

            status_code = http_row.get('status_code', None) if http_row.get('status_code', None) else None
            status_msg = http_row.get('status_msg', None) if http_row.get('status_msg', None) else None
            req_types = http_row.get('orig_mime_types', [])
            resp_types = http_row.get('resp_mime_types', [])

            parsed_log_line = {
                'timestamp': connection.get('ts', 0),
                'useragent': http_row.get('user_agent', ''),
                'domain': domain,
                'referrer': http_row.get('referrer', ''),
                'uri': http_row.get('uri', ''),
                'method': http_row.get('method', ''),
                'http_version': http_row.get('version', ''),
                'src_ip': http_row.get('id.orig_h', ''),
                'src_port': http_row.get('id.orig_p', ''),
                'dst_ip': http_row.get('id.resp_h', ''),
                'dst_port': http_row.get('id.resp_p', ''),
                'client_bytes': connection['orig_bytes'],
                'server_bytes': connection['resp_bytes'],
                'time_taken_ms': connection['duration'] * 1000.0,
                'service': connection.get('service', ''),
                'status_code': status_code,
                'status_msg': status_msg,
                'req_types': req_types,
                'resp_types': resp_types,

                # TODO: these do not seem like files - need to filter them better if NSKP output is actual files

                'file_types': sorted(list({c.get('mime_type', '') for c in files} if files else set())),
                'file_sizes': sorted(list({c.get('total_bytes', 0) for c in files} if files else set())),
                'file_hashes': sorted(list({c.get('md5', '') for c in files} if files else set()))
            }

            parsed_result.append(parsed_log_line)

        except Exception as e:
            logger.error("[!!] Running into error", str(e))

    save_json_data(parsed_result, output_path)
