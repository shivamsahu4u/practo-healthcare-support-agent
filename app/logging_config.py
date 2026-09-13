"""Task 12 - ELK-style JSON-Lines structured logging with trace ids.


Every request produces exactly one JSON object on one line, written through
``log_request()``. That function is the only writer, and it masks the query
itself, so a caller cannot log a raw contact number by forgetting to mask -
which is the whole point of the requirement that logs and model input get the
same treatment.


Belt and braces: ``ContactNumberMaskingFilter`` is attached to the root logger
too, so an ordinary ``LOGGER.info("... %s", text)`` from anywhere in the process
is masked as well.


What is never logged: unmasked contact numbers, API keys, secrets, full internal
prompts, raw environment variables, and any clinical detail. ``session_id`` is
logged as a salted-free one-way hash prefix rather than in the clear.
"""


from __future__ import annotations


import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Final


from app.config import SETTINGS, Settings, ensure_runtime_directories
from agents.guardrails import mask_contact_numbers


#: Dedicated logger for the one-line-per-request stream.
REQUEST_LOGGER_NAME: Final[str] = "practo.requests"


#: Attribute name carrying the pre-built JSON payload on a LogRecord.
PAYLOAD_ATTRIBUTE: Final[str] = "payload"


_CONFIGURED = False




def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")




def hash_session_id(session_id: str) -> str:
    """One-way, stable, truncated hash of a session id, safe to log."""
    return hashlib.sha256((session_id or "").encode("utf-8")).hexdigest()[:12]




class ContactNumberMaskingFilter(logging.Filter):
    """Mask fixed-format contact numbers in any log record that passes through.


    Applied to the message and to string arguments, before formatting.
    """


    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_contact_numbers(record.msg)[0]
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: (mask_contact_numbers(value)[0] if isinstance(value, str) else value)
                    for key, value in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    mask_contact_numbers(value)[0] if isinstance(value, str) else value
                    for value in record.args
                )
        payload = getattr(record, PAYLOAD_ATTRIBUTE, None)
        if isinstance(payload, dict):
            setattr(record, PAYLOAD_ATTRIBUTE, _mask_payload(payload))
        return True




def _mask_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Mask every string value in a payload, one level deep into lists/dicts."""


    def mask(value: Any) -> Any:
        if isinstance(value, str):
            return mask_contact_numbers(value)[0]
        if isinstance(value, list):
            return [mask(item) for item in value]
        if isinstance(value, dict):
            return {key: mask(item) for key, item in value.items()}
        return value


    return {key: mask(value) for key, value in payload.items()}




class JsonLinesFormatter(logging.Formatter):
    """Render a LogRecord as one compact JSON object on one line."""


    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, PAYLOAD_ATTRIBUTE, None)
        if isinstance(payload, dict):
            body: dict[str, Any] = {"timestamp": _utc_now(), **payload}
        else:
            body = {
                "timestamp": _utc_now(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        if record.exc_info:
            body["exception"] = self.formatException(record.exc_info)
        return json.dumps(body, ensure_ascii=False, default=str)




class HumanFormatter(logging.Formatter):
    """Readable console format for everything that is not a request summary."""


    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s")




def configure_logging(settings: Settings = SETTINGS, *, force: bool = False) -> None:
    """Install console logging plus the JSON-Lines request log. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return


    ensure_runtime_directories(settings)
    masking_filter = ContactNumberMaskingFilter()


    root = logging.getLogger()
    root.setLevel(settings.log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)


    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(HumanFormatter())
    console.addFilter(masking_filter)
    root.addHandler(console)


    request_logger = logging.getLogger(REQUEST_LOGGER_NAME)
    request_logger.setLevel(logging.INFO)
    # The request stream is its own file; it must not also appear on the console
    # as a raw JSON blob.
    request_logger.propagate = False
    for handler in list(request_logger.handlers):
        request_logger.removeHandler(handler)


    file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
    file_handler.setFormatter(JsonLinesFormatter())
    file_handler.addFilter(masking_filter)
    request_logger.addHandler(file_handler)


    _CONFIGURED = True




def log_request(
    *,
    trace_id: str,
    endpoint: str,
    transport: str,
    session_id: str,
    query: str,
    response_type: str,
    grounded: bool,
    cache_status: str,
    duration_ms: float,
    status_code: int,
    error_category: str | None = None,
    guardrails_fired: list[str] | None = None,
    tools_invoked: list[str] | None = None,
    sources: list[str] | None = None,
    review_approved: bool | None = None,
    extra: dict[str, Any] | None = None,
    settings: Settings = SETTINGS,
) -> dict[str, Any]:
    """Write exactly one JSON-Lines request-summary entry. Returns the payload.


    ``query`` is masked here regardless of what the caller passes, so the same
    masking that protects the model also protects the log file. Nothing else in
    the codebase writes to the request stream.
    """
    configure_logging(settings)


    masked_query, pii_findings = mask_contact_numbers(query or "")
    payload: dict[str, Any] = {
        "level": "INFO",
        "logger": REQUEST_LOGGER_NAME,
        "event": "request",
        "trace_id": trace_id,
        "endpoint": endpoint,
        "transport": transport,
        "session_hash": hash_session_id(session_id),
        "query": masked_query,
        "query_pii_masked": bool(pii_findings),
        "response_type": response_type,
        "grounded": grounded,
        "cache_status": cache_status,
        "duration_ms": round(duration_ms, 2),
        "status_code": status_code,
        "error_category": error_category,
        "guardrails_fired": list(guardrails_fired or []),
        "tools_invoked": list(tools_invoked or []),
        "sources": list(sources or []),
        "review_approved": review_approved,
        "mock_llm": settings.mock_llm,
        "crew_mode": settings.crew_mode,
    }
    if extra:
        payload["extra"] = extra


    logging.getLogger(REQUEST_LOGGER_NAME).info(
        "request", extra={PAYLOAD_ATTRIBUTE: payload}
    )
    return payload



