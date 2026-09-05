"""Unit tests for semantic_search.s3_search_log_uploader.

Mirrors the mocking style used for boto3-backed uploaders in this package:
patch `boto3.client` at the module namespace, never hit real AWS.
"""
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

from semantic_search import s3_search_log_uploader as uploader


@pytest.fixture(autouse=True)
def _clear_s3_env(monkeypatch):
    monkeypatch.delenv('S3_PRETRAINED_DIR', raising=False)
    monkeypatch.delenv('AWS_REGION', raising=False)
    monkeypatch.delenv('AWS_DEFAULT_REGION', raising=False)


class TestBucketFromS3Uri:
    def test_valid_uri(self):
        assert uploader._bucket_from_s3_uri('s3://my-bucket/prefix/') == 'my-bucket'

    def test_valid_uri_no_prefix(self):
        assert uploader._bucket_from_s3_uri('s3://my-bucket') == 'my-bucket'

    def test_non_s3_uri(self):
        assert uploader._bucket_from_s3_uri('/local/path') is None

    def test_empty(self):
        assert uploader._bucket_from_s3_uri('') is None


class TestIsKatanaEnv:
    def test_true_when_s3_uri_set(self, monkeypatch):
        monkeypatch.setenv('S3_PRETRAINED_DIR', 's3://bucket/models')
        assert uploader.is_katana_env() is True

    def test_false_when_unset(self):
        assert uploader.is_katana_env() is False

    def test_false_when_local_path(self, monkeypatch):
        monkeypatch.setenv('S3_PRETRAINED_DIR', '/local/models')
        assert uploader.is_katana_env() is False


class TestUploadSearchResultJson:
    def test_skips_when_no_bucket_configured(self):
        with patch('semantic_search.s3_search_log_uploader.boto3.client') as mock_client:
            uploader.upload_search_result_json({'query': 'q'}, 'req-1', 1_700_000_000.0)
            mock_client.assert_not_called()

    def test_writes_datewise_partitioned_key(self, monkeypatch):
        monkeypatch.setenv('S3_PRETRAINED_DIR', 's3://my-bucket/models')
        mock_s3 = MagicMock()
        with patch('semantic_search.s3_search_log_uploader.boto3.client', return_value=mock_s3):
            # 2024-03-05T00:00:00Z == 1709596800
            uploader.upload_search_result_json(
                {'query': 'expiring .com', 'answer_mode': 'search'}, 'req-42', 1_709_596_800.0
            )
        mock_s3.put_object.assert_called_once()
        _, kwargs = mock_s3.put_object.call_args
        assert kwargs['Bucket'] == 'my-bucket'
        assert kwargs['Key'] == 'search_logs/2024/03/05/req-42.json'
        assert kwargs['ContentType'] == 'application/json'
        assert b'expiring .com' in kwargs['Body']

    def test_swallows_client_error(self, monkeypatch):
        monkeypatch.setenv('S3_PRETRAINED_DIR', 's3://my-bucket/models')
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'nope'}}, 'PutObject'
        )
        with patch('semantic_search.s3_search_log_uploader.boto3.client', return_value=mock_s3):
            uploader.upload_search_result_json({'query': 'q'}, 'req-1', 1_709_596_800.0)
        # Must not raise — errors degrade to a logged warning only.


class TestEnsureSearchLogsRetentionPolicy:
    def test_skips_when_no_bucket_configured(self):
        with patch('semantic_search.s3_search_log_uploader.boto3.client') as mock_client:
            uploader.ensure_search_logs_retention_policy()
            mock_client.assert_not_called()

    def test_creates_rule_when_no_existing_lifecycle_config(self, monkeypatch):
        monkeypatch.setenv('S3_PRETRAINED_DIR', 's3://my-bucket/models')
        mock_s3 = MagicMock()
        mock_s3.get_bucket_lifecycle_configuration.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': 'NoSuchLifecycleConfiguration', 'Message': 'none'}},
            'GetBucketLifecycleConfiguration',
        )
        with patch('semantic_search.s3_search_log_uploader.boto3.client', return_value=mock_s3):
            uploader.ensure_search_logs_retention_policy()
        mock_s3.put_bucket_lifecycle_configuration.assert_called_once()
        _, kwargs = mock_s3.put_bucket_lifecycle_configuration.call_args
        rules = kwargs['LifecycleConfiguration']['Rules']
        assert len(rules) == 1
        assert rules[0]['ID'] == 'search-logs-3yr-expiration'
        assert rules[0]['Filter'] == {'Prefix': 'search_logs/'}
        assert rules[0]['Expiration'] == {'Days': 1095}

    def test_preserves_other_existing_rules_and_replaces_own(self, monkeypatch):
        monkeypatch.setenv('S3_PRETRAINED_DIR', 's3://my-bucket/models')
        mock_s3 = MagicMock()
        mock_s3.get_bucket_lifecycle_configuration.return_value = {
            'Rules': [
                {'ID': 'feedback-retention', 'Filter': {'Prefix': 'feedback/'},
                 'Status': 'Enabled', 'Expiration': {'Days': 30}},
                {'ID': 'search-logs-3yr-expiration', 'Filter': {'Prefix': 'search_logs/'},
                 'Status': 'Enabled', 'Expiration': {'Days': 1}},  # stale — must be replaced
            ]
        }
        with patch('semantic_search.s3_search_log_uploader.boto3.client', return_value=mock_s3):
            uploader.ensure_search_logs_retention_policy()
        _, kwargs = mock_s3.put_bucket_lifecycle_configuration.call_args
        rules = kwargs['LifecycleConfiguration']['Rules']
        ids = {r['ID']: r for r in rules}
        assert set(ids) == {'feedback-retention', 'search-logs-3yr-expiration'}
        assert ids['search-logs-3yr-expiration']['Expiration'] == {'Days': 1095}

    def test_swallows_client_error(self, monkeypatch):
        monkeypatch.setenv('S3_PRETRAINED_DIR', 's3://my-bucket/models')
        mock_s3 = MagicMock()
        mock_s3.get_bucket_lifecycle_configuration.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'nope'}},
            'GetBucketLifecycleConfiguration',
        )
        with patch('semantic_search.s3_search_log_uploader.boto3.client', return_value=mock_s3):
            uploader.ensure_search_logs_retention_policy()
        # Must not raise — missing IAM permission degrades to a logged warning only.
