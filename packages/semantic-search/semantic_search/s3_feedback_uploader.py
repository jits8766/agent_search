"""S3 feedback uploader for Katana (cloud) deployments.

When S3_PRETRAINED_DIR points to an s3:// URI the service is running in a
cloud/Katana environment that has S3 access.  Each ``uat_feedback`` signal is
written as a single-row CSV to the *parent bucket* of S3_PRETRAINED_DIR under
the ``feedback/YYYY/MM/DD/{signal_id}.csv`` key so records stay queryable via
Athena partitioned scans.

Parent-bucket derivation:
  S3_PRETRAINED_DIR = s3://gd-auctions-dev-private-us-west-2/pretrained/
  bucket            = gd-auctions-dev-private-us-west-2
  key prefix        = feedback/

Fire-and-forget: all errors are logged as warnings; nothing is raised.
"""
import csv
import datetime
import io
import os
from typing import Optional

import boto3
import botocore.exceptions

from semantic_search.contracts import FeedbackSignal
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_CSV_FIELDS = [
    'signal_id',
    'search_id',
    'request_id',
    'signal_type',
    'signal_origin',
    'created_at_iso',
    'comment',
    'query',
]


def _bucket_from_s3_uri(uri: str) -> Optional[str]:
    """Return bucket name from s3://bucket/... or None."""
    if not uri.startswith('s3://'):
        return None
    bucket = uri[5:].split('/')[0]
    return bucket or None


def is_katana_env() -> bool:
    """True when S3_PRETRAINED_DIR is an s3:// URI — indicates cloud/Katana deployment."""
    return os.environ.get('S3_PRETRAINED_DIR', '').startswith('s3://')


def upload_feedback_csv(signal: FeedbackSignal) -> None:
    """Upload signal as a single-row CSV to S3. Logs errors, never raises."""
    s3_pretrained = os.environ.get('S3_PRETRAINED_DIR', '')
    bucket = _bucket_from_s3_uri(s3_pretrained)
    if not bucket:
        logger.warning(f's3_feedback_upload_skipped reason=no_bucket signal_id={signal.signal_id}')
        return

    dt = datetime.datetime.utcfromtimestamp(signal.created_at)
    key = f"feedback/{dt.strftime('%Y/%m/%d')}/{signal.signal_id}.csv"

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    writer.writerow({
        'signal_id': signal.signal_id,
        'search_id': signal.search_id,
        'request_id': signal.request_id,
        'signal_type': signal.signal_type,
        'signal_origin': signal.signal_origin,
        'created_at_iso': dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'comment': signal.payload.get('comment', ''),
        'query': signal.payload.get('query', '') or '',
    })
    body = buf.getvalue().encode('utf-8')

    try:
        region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or ''
        s3 = boto3.client('s3', region_name=region or None)
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType='text/csv')
        logger.info(f's3_feedback_uploaded bucket={bucket} key={key} signal_id={signal.signal_id}')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as exc:
        logger.warning(
            f's3_feedback_upload_failed signal_id={signal.signal_id} '
            f'bucket={bucket} key={key} error_type={type(exc).__name__} error={exc}'
        )


def read_feedback_from_s3(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list:
    """Read all feedback CSVs from S3 and return merged list of dicts.

    :param date_from: ISO date string 'YYYY-MM-DD' — skip objects before this date
    :param date_to:   ISO date string 'YYYY-MM-DD' — skip objects after this date
    :return: list of row dicts sorted by created_at_iso ascending
    :raises RuntimeError: when S3_PRETRAINED_DIR is not an s3:// URI
    """
    s3_pretrained = os.environ.get('S3_PRETRAINED_DIR', '')
    bucket = _bucket_from_s3_uri(s3_pretrained)
    if not bucket:
        raise RuntimeError('S3_PRETRAINED_DIR is not set or is not an s3:// URI')

    region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or ''
    s3 = boto3.client('s3', region_name=region or None)

    prefix = 'feedback/'
    paginator = s3.get_paginator('list_objects_v2')
    rows: list = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key: str = obj['Key']
            if not key.endswith('.csv'):
                continue
            # key format: feedback/YYYY/MM/DD/{signal_id}.csv
            parts = key.split('/')
            if len(parts) >= 4:
                obj_date = f'{parts[1]}-{parts[2]}-{parts[3]}'
                if date_from and obj_date < date_from:
                    continue
                if date_to and obj_date > date_to:
                    continue
            try:
                resp = s3.get_object(Bucket=bucket, Key=key)
                body = resp['Body'].read().decode('utf-8')
                reader = csv.DictReader(io.StringIO(body))
                rows.extend(list(reader))
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as exc:
                logger.warning(f's3_feedback_read_failed key={key} error_type={type(exc).__name__} error={exc}')

    rows.sort(key=lambda r: r.get('created_at_iso', ''), reverse=True)
    logger.info(f's3_feedback_read_complete bucket={bucket} count={len(rows)} date_from={date_from} date_to={date_to}')
    return rows


__all__ = ['is_katana_env', 'upload_feedback_csv', 'read_feedback_from_s3']
