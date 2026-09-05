"""S3 search-log uploader for Katana (cloud) deployments.

Mirrors ``s3_feedback_uploader.py``: when S3_PRETRAINED_DIR points to an s3:// URI, each
completed /search execution is written as a single JSON object to the *parent bucket* of
S3_PRETRAINED_DIR under the ``search_logs/YYYY/MM/DD/{request_id}.json`` key, so records
stay queryable via Athena partitioned scans and support later log analysis / dashboarding.

Retention: an S3 Lifecycle Expiration rule scoped to the ``search_logs/`` prefix bounds
storage to 3 years (1095 days) — objects older than that are auto-deleted by S3 itself.
No custom scan-and-delete code runs in the request path or on a schedule; the rule is
applied (idempotently) once at service startup via ``ensure_search_logs_retention_policy``.

Fire-and-forget: all errors are logged as warnings; nothing is raised.
"""
import datetime
import json
import os
from typing import Any, Dict, Optional

import boto3
import botocore.exceptions

from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = 'search_logs/'
_RETENTION_DAYS = 1095  # 3 years
_RETENTION_RULE_ID = 'search-logs-3yr-expiration'


def _bucket_from_s3_uri(uri: str) -> Optional[str]:
    """Return bucket name from s3://bucket/... or None."""
    if not uri.startswith('s3://'):
        return None
    bucket = uri[5:].split('/')[0]
    return bucket or None


def is_katana_env() -> bool:
    """True when S3_PRETRAINED_DIR is an s3:// URI — indicates cloud/Katana deployment."""
    return os.environ.get('S3_PRETRAINED_DIR', '').startswith('s3://')


def upload_search_result_json(record: Dict[str, Any], request_id: str, created_at: float) -> None:
    """Upload a search result record as JSON to S3. Logs errors, never raises.
    :param record: Dict[str, Any] - JSON-serializable search result/dashboard record
    :param request_id: str - Request identifier, used as the object filename
    :param created_at: float - Unix timestamp used for the datewise partition
    """
    s3_pretrained = os.environ.get('S3_PRETRAINED_DIR', '')
    bucket = _bucket_from_s3_uri(s3_pretrained)
    if not bucket:
        logger.warning(f's3_search_log_upload_skipped reason=no_bucket request_id={request_id}')
        return

    dt = datetime.datetime.fromtimestamp(created_at, tz=datetime.timezone.utc)
    key = f"{_KEY_PREFIX}{dt.strftime('%Y/%m/%d')}/{request_id}.json"
    body = json.dumps(record, default=str).encode('utf-8')

    try:
        region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or ''
        s3 = boto3.client('s3', region_name=region or None)
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType='application/json')
        logger.info(f's3_search_log_uploaded bucket={bucket} key={key} request_id={request_id}')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as exc:
        logger.warning(
            f's3_search_log_upload_failed request_id={request_id} '
            f'bucket={bucket} key={key} error_type={type(exc).__name__} error={exc}'
        )


def ensure_search_logs_retention_policy(bucket: Optional[str] = None) -> None:
    """Ensure an S3 Lifecycle Expiration rule bounds `search_logs/` to 3 years.

    Best-effort and idempotent — merges a single rule (`_RETENTION_RULE_ID`) into the
    bucket's existing lifecycle configuration, preserving any other rules already present
    (e.g. the `feedback/` prefix, if one exists). Call once at startup; never called from
    the request path. Missing IAM permission (s3:GetBucketLifecycleConfiguration /
    s3:PutBucketLifecycleConfiguration) or no bucket configured both degrade to a logged
    warning — the service boots and search logging continues either way, just without the
    auto-expiry guarantee until an operator grants the permission or applies the rule
    through infra tooling instead.
    """
    if bucket is None:
        bucket = _bucket_from_s3_uri(os.environ.get('S3_PRETRAINED_DIR', ''))
    if not bucket:
        logger.warning('s3_search_log_retention_policy_skipped reason=no_bucket')
        return

    try:
        region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or ''
        s3 = boto3.client('s3', region_name=region or None)
        try:
            existing = s3.get_bucket_lifecycle_configuration(Bucket=bucket)
            rules = [r for r in existing.get('Rules', []) if r.get('ID') != _RETENTION_RULE_ID]
        except botocore.exceptions.ClientError as exc:
            if exc.response.get('Error', {}).get('Code') != 'NoSuchLifecycleConfiguration':
                raise
            rules = []
        rules.append({
            'ID': _RETENTION_RULE_ID,
            'Filter': {'Prefix': _KEY_PREFIX},
            'Status': 'Enabled',
            'Expiration': {'Days': _RETENTION_DAYS},
        })
        s3.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={'Rules': rules})
        logger.info(f's3_search_log_retention_policy_applied bucket={bucket} prefix={_KEY_PREFIX} days={_RETENTION_DAYS}')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as exc:
        logger.warning(
            f's3_search_log_retention_policy_failed bucket={bucket} '
            f'error_type={type(exc).__name__} error={exc}'
        )


__all__ = ['is_katana_env', 'upload_search_result_json', 'ensure_search_logs_retention_policy']
