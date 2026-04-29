"""
Unit tests for the Azure AI Search client (index document preparation and
split_field helper) – no live Azure credentials required.

The search query functions (search_rca_index, search_tsg_index) require an
active Azure AI Search service and are tested via the integration test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestODataEscape:
    def test_safe_string_unchanged(self):
        from pf_recovery_agent.search.azure_search_client import _odata_escape

        assert _odata_escape("PFGateway") == "PFGateway"

    def test_single_quote_doubled(self):
        from pf_recovery_agent.search.azure_search_client import _odata_escape

        assert _odata_escape("O'Brien") == "O''Brien"

    def test_multiple_quotes_escaped(self):
        from pf_recovery_agent.search.azure_search_client import _odata_escape

        assert _odata_escape("it's a 'test'") == "it''s a ''test''"

    def test_empty_string(self):
        from pf_recovery_agent.search.azure_search_client import _odata_escape

        assert _odata_escape("") == ""

    def test_comma_split(self):
        from pf_recovery_agent.search.azure_search_client import _split_field

        assert _split_field("a, b, c") == ["a", "b", "c"]

    def test_pipe_split(self):
        from pf_recovery_agent.search.azure_search_client import _split_field

        assert _split_field("step one | step two | step three", delimiter="|") == [
            "step one",
            "step two",
            "step three",
        ]

    def test_empty_string_returns_empty_list(self):
        from pf_recovery_agent.search.azure_search_client import _split_field

        assert _split_field("") == []

    def test_single_item(self):
        from pf_recovery_agent.search.azure_search_client import _split_field

        assert _split_field("PFGateway") == ["PFGateway"]

    def test_filters_empty_items(self):
        from pf_recovery_agent.search.azure_search_client import _split_field

        assert _split_field("a,,b,") == ["a", "b"]


class TestIndexerDocConversion:
    """Test that RCA/TSG/dependency dicts are correctly transformed into index docs."""

    def _sample_rca(self):
        return {
            "id": "RCA-TEST-001",
            "title": "Test RCA",
            "affected_services": ["ServiceA", "ServiceB"],
            "date": "2024-01-01",
            "root_cause": "A thing broke",
            "mitigation_steps": ["Step 1", "Step 2"],
            "time_to_mitigate_minutes": 30,
            "tags": ["tag1", "tag2"],
            "lessons_learned": "Don't do that",
        }

    def _sample_tsg(self):
        return {
            "id": "TSG-TEST-001",
            "title": "Test TSG",
            "applicable_services": ["ServiceA"],
            "symptoms": ["symptom A", "symptom B"],
            "diagnostic_steps": ["diag 1", "diag 2"],
            "mitigation_steps": ["mit 1", "mit 2"],
            "escalation_path": "Team A -> Team B",
            "tags": ["tag1"],
        }

    def _sample_dep(self):
        return {
            "service_name": "ServiceA",
            "cluster": "Cluster-East",
            "tier": 1,
            "description": "A critical service",
            "upstream_dependencies": ["AAD", "DNS"],
            "downstream_dependents": ["ServiceB", "ServiceC"],
        }

    def test_rca_doc_has_stringified_lists(self):
        from pf_recovery_agent.search.indexer import _rca_to_doc

        doc = _rca_to_doc(self._sample_rca(), None, None)
        assert doc["affected_services_str"] == "ServiceA, ServiceB"
        assert "Step 1" in doc["mitigation_steps_str"]
        assert doc["tags_str"] == "tag1, tag2"

    def test_rca_doc_id_preserved(self):
        from pf_recovery_agent.search.indexer import _rca_to_doc

        doc = _rca_to_doc(self._sample_rca(), None, None)
        assert doc["id"] == "RCA-TEST-001"

    def test_rca_doc_no_vector_when_no_client(self):
        from pf_recovery_agent.search.indexer import _rca_to_doc

        doc = _rca_to_doc(self._sample_rca(), None, None)
        assert "content_vector" not in doc

    def test_tsg_doc_has_pipe_separated_steps(self):
        from pf_recovery_agent.search.indexer import _tsg_to_doc

        doc = _tsg_to_doc(self._sample_tsg(), None, None)
        assert " | " in doc["diagnostic_steps_str"]
        assert " | " in doc["mitigation_steps_str"]

    def test_tsg_doc_symptoms_joined(self):
        from pf_recovery_agent.search.indexer import _tsg_to_doc

        doc = _tsg_to_doc(self._sample_tsg(), None, None)
        assert "symptom A" in doc["symptoms_str"]

    def test_dep_doc_id_uses_service_and_cluster(self):
        from pf_recovery_agent.search.indexer import _dep_to_doc

        doc = _dep_to_doc(self._sample_dep())
        assert "ServiceA" in doc["id"]
        assert "Cluster-East" in doc["id"]

    def test_dep_doc_upstream_stringified(self):
        from pf_recovery_agent.search.indexer import _dep_to_doc

        doc = _dep_to_doc(self._sample_dep())
        assert "AAD" in doc["upstream_dependencies_str"]
        assert "DNS" in doc["upstream_dependencies_str"]

    def test_dep_doc_tier_preserved(self):
        from pf_recovery_agent.search.indexer import _dep_to_doc

        doc = _dep_to_doc(self._sample_dep())
        assert doc["tier"] == 1
