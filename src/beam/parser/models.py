"""
Contains Pydantic models for BEAM
"""

from pydantic import BaseModel

class NetskopeTransaction(BaseModel):
    """
    Pydantic Model for Netskope Transaction
    """
    timestamp: str
    day: str
    hour: str
    access_method: str
    useragent: str
    hostname: str
    referer: str | None  # Use the Union type to allow for a nullable string
    uri_scheme: str
    http_method: str
    http_status: str
    rs_status: str | None  # Allow null value
    ssl_ja3: str | None  # Allow null value
    ssl_ja3s: str | None  # Allow null value
    file_type: str | None  # Allow null value
    traffic_type: str
    client_http_version: str | None  # Allow null value
    srcport: str
    client_src_port: str | None  # Allow null value
    client_dst_port: str | None  # Allow null value
    client_connect_port: str | None  # Allow null value
    server_src_port: str | None  # Allow null value
    server_dst_port: str | None  # Allow null value
    req_content_type: str
    resp_content_type: str
    server_ssl_error: str | None  # Allow null value
    client_ssl_error: str | None  # Allow null value
    error: str | None  # Allow null value
    ssl_bypass: str | None  # Allow null value
    ssl_bypass_reason: str | None  # Allow null value
    ssl_fronting_error: str | None  # Allow null value
    time_taken_ms: str
    client_bytes: str
    server_bytes: str
    file_sha256: str | None  # Allow null value
    file_size: str | None  # Allow null value
    url: str
