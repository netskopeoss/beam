"""
Contains Pydantic models for BEAM
"""

from pydantic import BaseModel


class Transaction(BaseModel):
    """
    Pydantic Model for a transaction
    """
    timestamp: float
    useragent: str
    hostname: str
    domain: str
    uri_scheme: str
    http_method: str
    http_status: str
    client_http_version: str | None  # Allow null value
    req_content_type: str
    resp_content_type: str
    time_taken_ms: int
    client_bytes: float
    server_bytes: float
    referer: str | None  # Use the Union type to allow for a nullable string
    referer_domain: str | None  # Use the Union type to allow for a nullable string
    url: str
    uri: str
    src_ip: str
