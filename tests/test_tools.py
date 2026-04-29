"""Unit tests for local knowledge-base tools (no external services required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src package is importable when running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Dependency tool tests
# ---------------------------------------------------------------------------


class TestGetServiceDependencies:
    def test_known_service_returns_direct_info(self):
        from pf_recovery_agent.tools.dependency_tool import get_service_dependencies

        result = get_service_dependencies(["PFGateway"])
        assert "PFGateway" in result["direct"]
        info = result["direct"]["PFGateway"]
        assert "upstream_dependencies" in info
        assert "downstream_dependents" in info
        assert "tier" in info

    def test_blast_radius_excludes_input_services(self):
        from pf_recovery_agent.tools.dependency_tool import get_service_dependencies

        result = get_service_dependencies(["PFGateway"])
        assert "PFGateway" not in result["blast_radius"]

    def test_unknown_service_returns_error(self):
        from pf_recovery_agent.tools.dependency_tool import get_service_dependencies

        result = get_service_dependencies(["NonExistentService"])
        assert "error" in result["direct"]["NonExistentService"]

    def test_multiple_services_blast_radius(self):
        from pf_recovery_agent.tools.dependency_tool import get_service_dependencies

        result = get_service_dependencies(["PFGateway", "PFOrchestrator"])
        # Both services' upstreams/downstreams should expand the blast radius
        assert len(result["blast_radius"]) > 0

    def test_tier1_risks_are_subset_of_blast_radius(self):
        from pf_recovery_agent.tools.dependency_tool import get_service_dependencies

        result = get_service_dependencies(["PFWorkerPool"])
        tier1 = set(result["tier1_risks"])
        blast = set(result["blast_radius"])
        assert tier1.issubset(blast)


# ---------------------------------------------------------------------------
# RCA tool tests (local fallback path)
# ---------------------------------------------------------------------------


class TestSearchRcasLocal:
    def _search(self, services, keywords=None, max_results=5):
        # Always use the local fallback for unit tests
        from pf_recovery_agent.tools.rca_tool import _search_rcas_local

        return _search_rcas_local(services, keywords, max_results)

    def test_returns_list(self):
        results = self._search(["PFGateway"])
        assert isinstance(results, list)

    def test_service_match_returns_results(self):
        results = self._search(["PFGateway"])
        assert len(results) > 0

    def test_keyword_improves_relevance(self):
        results_no_kw = self._search(["PFGateway"])
        results_with_kw = self._search(["PFGateway"], keywords=["tls", "certificate"])
        # TLS-tagged RCA should rank first with keywords
        assert results_with_kw[0]["id"] == "RCA-2024-001"

    def test_max_results_respected(self):
        results = self._search(["PFGateway", "PFOrchestrator", "PFStorageService"], max_results=2)
        assert len(results) <= 2

    def test_no_match_returns_empty(self):
        results = self._search(["UnknownServiceXYZ"])
        assert results == []

    def test_result_fields_present(self):
        results = self._search(["PFStorageService"])
        assert len(results) > 0
        rca = results[0]
        for field in ["id", "title", "root_cause", "mitigation_steps", "time_to_mitigate_minutes", "relevance_score"]:
            assert field in rca, f"Field '{field}' missing from RCA result"

    def test_mitigation_steps_is_list(self):
        results = self._search(["PFOrchestrator"])
        assert len(results) > 0
        assert isinstance(results[0]["mitigation_steps"], list)

    def test_relevance_score_descending(self):
        results = self._search(["PFGateway", "PFOrchestrator"])
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# TSG tool tests (local fallback path)
# ---------------------------------------------------------------------------


class TestSearchTsgsLocal:
    def _search(self, services, symptoms=None, max_results=5):
        from pf_recovery_agent.tools.tsg_tool import _search_tsgs_local

        return _search_tsgs_local(services, symptoms, max_results)

    def test_returns_list(self):
        results = self._search(["PFGateway"])
        assert isinstance(results, list)

    def test_gateway_returns_tsg001(self):
        results = self._search(["PFGateway"])
        ids = [r["id"] for r in results]
        assert "TSG-001" in ids

    def test_orchestrator_returns_tsg002(self):
        results = self._search(["PFOrchestrator"])
        ids = [r["id"] for r in results]
        assert "TSG-002" in ids

    def test_symptom_improves_score(self):
        no_symptom = self._search(["PFWorkerPool"])
        with_symptom = self._search(["PFWorkerPool"], symptoms=["service-bus", "dead-letter"])
        # TSG-004 should appear with symptoms
        ids_with = [r["id"] for r in with_symptom]
        assert "TSG-004" in ids_with

    def test_max_results_respected(self):
        results = self._search(["PFGateway", "PFOrchestrator"], max_results=1)
        assert len(results) <= 1

    def test_result_fields_present(self):
        results = self._search(["PFGateway"])
        assert len(results) > 0
        tsg = results[0]
        for field in ["id", "title", "diagnostic_steps", "mitigation_steps", "escalation_path", "relevance_score"]:
            assert field in tsg, f"Field '{field}' missing from TSG result"

    def test_diagnostic_steps_is_list(self):
        results = self._search(["PFOrchestrator"])
        assert isinstance(results[0]["diagnostic_steps"], list)

    def test_no_match_returns_empty(self):
        results = self._search(["CompletelyUnknownService"])
        assert results == []
