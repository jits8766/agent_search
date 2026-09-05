"""Tests for the user search history store.

Coverage matrix (per ``testing.mdc`` §7):

``UserSearchHistoryStore`` (record/list/delete + opt-in/opt-out state machine):
- record_and_list                               -> TestUserSearchHistoryStore::test_record_and_list
- empty_user_id_rejected_on_record              -> TestUserSearchHistoryStore::test_empty_user_id_rejected_on_record
- empty_user_id_rejected_on_optout              -> TestUserSearchHistoryStore::test_empty_user_id_rejected_on_optout
- query_truncated_to_max_length                 -> TestUserSearchHistoryStore::test_query_truncated_to_max_length
- max_entries_enforced                          -> TestUserSearchHistoryStore::test_max_entries_enforced
- retention_window_prunes_old_entries           -> TestUserSearchHistoryStore::test_retention_window_prunes_old_entries
- repeat_query_count_within_window              -> TestUserSearchHistoryStore::test_repeat_query_count_within_window
- optout_blocks_writes_and_purges_entries       -> TestUserSearchHistoryStore::test_optout_blocks_writes_and_purges_entries
- optin_reverses_optout                         -> TestUserSearchHistoryStore::test_optin_reverses_optout
- delete_user_removes_all_entries               -> TestUserSearchHistoryStore::test_delete_user_removes_all_entries
"""
import time

import pytest

from semantic_search.config.models import AgentSearchConfig
from semantic_search.core.exceptions import HistoryError
from semantic_search.history.store import UserSearchHistoryStore


class TestUserSearchHistoryStore:
    def test_record_and_list(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        store.record('alice', 'tech startup names', 'explore', ['a', 'b'])
        entries = store.list_entries('alice')
        assert len(entries) == 1
        assert entries[0].normalized_query == 'tech startup names'

    def test_max_entries_enforced(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        cap = config.history.max_entries_per_user
        for i in range(cap + 5):
            store.record('alice', f'query {i}', 'explore', [])
        assert len(store.list_entries('alice')) == cap

    def test_query_truncated_to_max_length(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        long_q = 'x' * (config.history.max_query_length * 2)
        store.record('alice', long_q, 'explore', [])
        entries = store.list_entries('alice')
        assert len(entries[0].normalized_query) == config.history.max_query_length

    def test_optout_blocks_writes_and_purges_entries(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        store.record('alice', 'q1', 'hybrid', [])
        store.opt_out('alice')
        assert store.is_opted_out('alice') is True
        assert store.list_entries('alice') == []
        assert store.record('alice', 'q2', 'hybrid', []) is None

    def test_optin_reverses_optout(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        store.opt_out('alice')
        store.opt_in('alice')
        assert store.is_opted_out('alice') is False
        e = store.record('alice', 'q3', 'hybrid', [])
        assert e is not None

    def test_empty_user_id_rejected_on_record(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        with pytest.raises(HistoryError):
            store.record('', 'q', 'hybrid', [])

    def test_empty_user_id_rejected_on_optout(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        with pytest.raises(HistoryError):
            store.opt_out('')

    def test_delete_user_removes_all_entries(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        for i in range(3):
            store.record('alice', f'q{i}', 'hybrid', [])
        removed = store.delete_user('alice')
        assert removed == 3
        assert store.list_entries('alice') == []

    def test_repeat_query_count_within_window(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        for _ in range(3):
            store.record('alice', 'expiring io domains', 'hybrid', [])
        store.record('alice', 'something different', 'hybrid', [])
        assert store.repeat_query_count('alice', 'expiring io domains', 86400.0) == 3
        assert store.repeat_query_count('alice', 'unseen', 86400.0) == 0

    def test_retention_window_prunes_old_entries(self, config: AgentSearchConfig):
        store = UserSearchHistoryStore(config.history)
        e = store.record('alice', 'old query', 'hybrid', [])
        assert e is not None
        # Force entry to look ancient
        e.created_at = time.time() - (config.history.retention_days * 86400.0) - 60.0
        # The deque still has the entry until list_entries prunes it
        store._entries['alice'][0] = e  # noqa: SLF001 — internal mutation for the test
        assert store.list_entries('alice') == []
