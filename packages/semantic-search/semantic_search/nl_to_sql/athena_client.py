"""Athena client for the semantic_search seed pipeline.

Executes SQL on AWS Athena, polls for completion, fetches the CSV result from
S3, and returns (rows, columns, latency_ms).

Interface consumed by db_seed_source / seed_merge:
  - credentials_available: bool
  - fetch_sql_async(query, timeout_seconds) -> (List[Dict], List[str], float)
  - execute_ddl(query, timeout_seconds) -> None
  - database_exists(database) -> bool  (Glue GetDatabase; skip CREATE when present)
"""
import asyncio
import csv
import io
import os
import time
from typing import Any, Dict, List, Tuple

import boto3
from botocore.config import Config

from semantic_search.config.nl_to_sql_models import AthenaClientConfig
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_GLUE_NOT_FOUND_CODES = frozenset({"EntityNotFoundException", "InvalidInputException"})
_TOKEN_EXPIRED_CODES = frozenset({"ExpiredTokenException", "ExpiredToken"})


class AthenaClient:
    """Athena client for seeding the in-memory vector/structured indexes.

    :param config: AthenaClientConfig - Typed config from nl_to_sql.athena in base.yaml
    """

    def __init__(self, config: AthenaClientConfig) -> None:
        if not isinstance(config, AthenaClientConfig):
            raise TypeError("AthenaClient requires an AthenaClientConfig")
        self._config = config
        self.region: str = config.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "")
        self.s3_output_location: str = config.s3_output_location or os.environ.get("S3_ATHENA_DIR", "")
        self.credentials_available: bool = False
        self._athena: Any = None
        self._s3: Any = None
        self._glue: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialise boto3 Athena, S3, and Glue clients. Sets credentials_available."""
        try:
            session = boto3.Session(region_name=self.region or None)
            role_arn = self._config.role_arn or os.environ.get("ATHENA_ROLE_ARN", "")
            if role_arn:
                sts = session.client("sts")
                assumed = sts.assume_role(RoleArn=role_arn, RoleSessionName="agent-search-athena")
                assumed_creds = assumed["Credentials"]
                session = boto3.Session(
                    aws_access_key_id=assumed_creds["AccessKeyId"],
                    aws_secret_access_key=assumed_creds["SecretAccessKey"],
                    aws_session_token=assumed_creds["SessionToken"],
                    region_name=self.region or None,
                )
                logger.info(f"athena_client assumed_role role_arn={role_arn}")
            creds = session.get_credentials()
            if creds is None:
                logger.warning("athena_client credentials=none")
                return
            frozen = creds.get_frozen_credentials()
            if not frozen.access_key:
                logger.warning("athena_client credentials=empty_access_key")
                return
            boto_cfg = Config(connect_timeout=self._config.connect_timeout, read_timeout=self._config.read_timeout, retries={"max_attempts": self._config.max_retries, "mode": "adaptive"})
            self._athena = session.client("athena")
            self._s3 = session.client("s3", config=boto_cfg)
            self._glue = session.client("glue")
            self.credentials_available = True
        except Exception as exc:
            logger.warning(f"athena_client init_failed error_type={type(exc).__name__}")

    async def fetch_sql_async(self, query: str, timeout_seconds: float) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """Execute SQL on Athena and return results as a list of row dicts.

        :param query: str - SQL to execute
        :param timeout_seconds: float - Wall-clock cap for the status poll loop
        :return: Tuple - (rows: List[Dict], columns: List[str], latency_ms: float)
        :raises RuntimeError: When credentials unavailable, S3 output not configured, or query fails
        """
        if not self.credentials_available:
            raise RuntimeError("athena_client credentials_unavailable")
        if not self.s3_output_location:
            raise RuntimeError("athena_client s3_output_location not configured — set nl_to_sql.athena.s3_output_location in config or S3_ATHENA_DIR env var")
        t0 = time.monotonic()
        output_location, exec_id = await self._execute_async(query, timeout_seconds)
        rows, columns = await self._fetch_csv_async(output_location)
        await self._delete_async(output_location)
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(f"athena_fetch_complete execution_id={exec_id} rows={len(rows)} latency_ms={latency_ms:.1f}")
        return rows, columns, round(latency_ms, 1)

    async def execute_ddl(self, query: str, timeout_seconds: float) -> None:
        """Execute a DDL statement (CREATE TABLE ... AS SELECT, DROP TABLE) on Athena.

        Unlike ``fetch_sql_async``, does not fetch or delete a result CSV — a
        DDL statement's output location holds a manifest, not row data.

        :param query: str - DDL SQL to execute
        :param timeout_seconds: float - Wall-clock cap for the status poll loop
        :raises RuntimeError: When credentials unavailable, S3 output not configured, or the statement fails
        """
        if not self.credentials_available:
            raise RuntimeError("athena_client credentials_unavailable")
        if not self.s3_output_location:
            raise RuntimeError("athena_client s3_output_location not configured — set nl_to_sql.athena.s3_output_location in config or S3_ATHENA_DIR env var")
        t0 = time.monotonic()
        _output_location, exec_id = await self._execute_async(query, timeout_seconds)
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(f"athena_ddl_complete execution_id={exec_id} latency_ms={latency_ms:.1f}")

    async def database_exists(self, database: str) -> bool:
        """Return True when the Glue/Athena database already exists.

        Uses ``glue:GetDatabase`` (not ``CREATE DATABASE``) so a pre-provisioned
        scratch DB can be reused under roles that allow Get* but deny CreateDatabase.

        :param database: str - Unquoted database / schema name
        :return: bool - True if Glue returns the database; False if not found
        :raises RuntimeError: When credentials unavailable or Glue call fails for a non-not-found reason
        """
        if not self.credentials_available or self._glue is None:
            raise RuntimeError("athena_client credentials_unavailable")
        if not isinstance(database, str) or not database.strip():
            raise RuntimeError("athena_client database_exists requires a non-empty database name")
        name = database.strip().lower()

        def _get() -> bool:
            try:
                self._glue.get_database(Name=name)
                return True
            except Exception as exc:
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
                if code in _GLUE_NOT_FOUND_CODES:
                    return False
                if code in _TOKEN_EXPIRED_CODES:
                    raise
                raise RuntimeError(
                    f"athena_client database_exists failed database={name} error_type={type(exc).__name__} code={code or 'unknown'}"
                ) from exc

        try:
            exists = await asyncio.to_thread(_get)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in _TOKEN_EXPIRED_CODES:
                logger.info(f"athena_client token_expired_on_get_database database={name} refreshing_credentials")
                self.refresh_credentials()
                if not self.credentials_available or self._glue is None:
                    raise RuntimeError("athena_client credential_refresh_failed") from exc
                exists = await asyncio.to_thread(_get)
            else:
                raise
        logger.info(f"athena_database_exists database={name} exists={exists}")
        return exists

    def refresh_credentials(self) -> None:
        """Re-run STS assume_role and reinitialise boto3 clients.

        Call when a request fails with ExpiredTokenException. Sets
        credentials_available=False on failure so callers can detect it.
        """
        self.credentials_available = False
        self._athena = None
        self._s3 = None
        self._glue = None
        self._init_clients()
        logger.info(f"athena_client credentials_refreshed available={self.credentials_available}")

    async def _execute_async(self, query: str, timeout_seconds: float) -> Tuple[str, str]:
        """Start an Athena query and poll until SUCCEEDED or timeout.

        :return: Tuple - (result_s3_path, execution_id)
        :raises RuntimeError: On FAILED status or timeout
        """
        execution_dt = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        output_prefix = self.s3_output_location.rstrip("/") + f"/athena-results/{execution_dt}/"
        try:
            response = await asyncio.to_thread(self._athena.start_query_execution, QueryString=query, ResultConfiguration={"OutputLocation": output_prefix})
        except Exception as _start_exc:
            _code = getattr(_start_exc, 'response', {}).get('Error', {}).get('Code', '')
            if _code in ('ExpiredTokenException', 'ExpiredToken'):
                logger.info("athena_client token_expired_on_start refreshing_credentials")
                self.refresh_credentials()
                if not self.credentials_available:
                    raise RuntimeError("athena_client credential_refresh_failed") from _start_exc
                response = await asyncio.to_thread(self._athena.start_query_execution, QueryString=query, ResultConfiguration={"OutputLocation": output_prefix})
            else:
                raise
        exec_id: str = response["QueryExecutionId"]
        deadline = time.monotonic() + timeout_seconds
        status = "RUNNING"
        status_response: Dict[str, Any] = {}
        while status in ("QUEUED", "RUNNING"):
            if time.monotonic() >= deadline:
                logger.warning(f"athena_query_timeout execution_id={exec_id} timeout_seconds={timeout_seconds}")
                raise RuntimeError(f"Athena query timed out after {timeout_seconds}s id={exec_id}")
            await asyncio.sleep(self._config.poll_sleep_seconds)
            try:
                status_response = await asyncio.to_thread(self._athena.get_query_execution, QueryExecutionId=exec_id)
            except Exception as _poll_exc:
                _code = getattr(_poll_exc, 'response', {}).get('Error', {}).get('Code', '')
                if _code in ('ExpiredTokenException', 'ExpiredToken'):
                    logger.info(f"athena_client token_expired_on_poll execution_id={exec_id} refreshing_credentials")
                    self.refresh_credentials()
                    if not self.credentials_available:
                        raise RuntimeError("athena_client credential_refresh_failed") from _poll_exc
                    status_response = await asyncio.to_thread(self._athena.get_query_execution, QueryExecutionId=exec_id)
                else:
                    raise
            status = status_response["QueryExecution"]["Status"]["State"]
        if status != "SUCCEEDED":
            reason = status_response.get("QueryExecution", {}).get("Status", {}).get("StateChangeReason", "unknown")
            raise RuntimeError(f"Athena query {status} id={exec_id}: {reason}")
        result_location: str = status_response["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
        return result_location, exec_id

    async def _fetch_csv_async(self, s3_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Download Athena CSV result from S3 and parse into row dicts.

        :param s3_path: str - s3://bucket/key
        :return: Tuple - (rows, column_names)
        """
        return await asyncio.to_thread(self._fetch_csv_sync, s3_path)

    def _fetch_csv_sync(self, s3_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Synchronous CSV fetch and parse.

        :param s3_path: str - s3://bucket/key
        :return: Tuple - (rows as dicts, column name list)
        """
        path = s3_path[5:] if s3_path.startswith("s3://") else s3_path
        bucket, key = path.split("/", 1)
        try:
            obj = self._s3.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            _code = getattr(exc, 'response', {}).get('Error', {}).get('Code', '')
            if _code in ('ExpiredTokenException', 'ExpiredToken'):
                logger.info("athena_client token_expired_on_fetch refreshing_credentials")
                self.refresh_credentials()
                if not self.credentials_available:
                    raise RuntimeError("athena_client credential_refresh_failed") from exc
                obj = self._s3.get_object(Bucket=bucket, Key=key)
            else:
                raise
        body = obj["Body"].read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(body))
        rows = list(reader)
        columns = list(reader.fieldnames or [])
        return rows, columns

    async def _delete_async(self, s3_path: str) -> None:
        """Best-effort S3 delete — swallows errors so callers are not blocked."""
        try:
            await asyncio.to_thread(self._delete_sync, s3_path)
        except Exception as exc:
            logger.warning(f"athena_s3_delete_failed path={s3_path} error={exc}")

    def _delete_sync(self, s3_path: str) -> None:
        """Synchronous S3 object delete."""
        path = s3_path[5:] if s3_path.startswith("s3://") else s3_path
        bucket, key = path.split("/", 1)
        try:
            self._s3.delete_object(Bucket=bucket, Key=key)
        except Exception as exc:
            _code = getattr(exc, 'response', {}).get('Error', {}).get('Code', '')
            if _code in ('ExpiredTokenException', 'ExpiredToken'):
                self.refresh_credentials()
                if self.credentials_available:
                    self._s3.delete_object(Bucket=bucket, Key=key)
                    return
            raise


__all__ = ["AthenaClient"]
