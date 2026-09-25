
from __future__ import annotations

from enum import StrEnum

class WorkerFailureReason(StrEnum):

    REGISTRY_AUTHENTICATION_FAILED = "registry_authentication_failed"
    REQUIRED_ASSET_UNAVAILABLE = "required_asset_unavailable"
    REQUEST_SCHEMA_INVALID = "request_schema_invalid"
    REQUEST_IDENTITY_DRIFT = "request_identity_drift"
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_NONZERO_EXIT = "worker_nonzero_exit"
    RESULT_MISSING = "result_missing"
    RESULT_SCHEMA_INVALID = "result_schema_invalid"
    UNKNOWN_OUTPUT_FIELD = "unknown_output_field"
    NONFINITE_METRIC = "nonfinite_metric"
    MISSING_REQUIRED_METRIC = "missing_required_metric"
    VECTOR_LENGTH_MISMATCH = "vector_length_mismatch"
    UNIT_CONTRACT_ERROR = "unit_contract_error"
    UPSTREAM_NUMERIC_SENTINEL = "upstream_numeric_sentinel"
    OUTPUT_IDENTITY_DRIFT = "output_identity_drift"
    EVIDENCE_PATH_VIOLATION = "evidence_path_violation"

class WorkerProtocolError(RuntimeError):

    def __init__(self, reason: WorkerFailureReason, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")

def protocol_error(reason: WorkerFailureReason, detail: str) -> WorkerProtocolError:

    return WorkerProtocolError(reason, detail)

__all__ = ("WorkerFailureReason", "WorkerProtocolError", "protocol_error")
