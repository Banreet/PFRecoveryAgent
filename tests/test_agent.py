"""Unit tests for the agent helper functions (no LLM calls required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestBuildUserMessage:
    def _make_outage(self, **kwargs):
        from datetime import datetime

        from pf_recovery_agent.models import OutageEvent, OutageSeverity

        defaults = {
            "id": "ICM-TEST-001",
            "title": "PFGateway failure",
            "affected_services": ["PFGateway"],
            "severity": OutageSeverity.SEV1,
            "start_time": datetime(2024, 1, 1, 12, 0, 0),
        }
        defaults.update(kwargs)
        return OutageEvent(**defaults)

    def test_message_contains_outage_id(self):
        from pf_recovery_agent.agent import _build_user_message

        outage = self._make_outage()
        msg = _build_user_message(outage)
        assert "ICM-TEST-001" in msg

    def test_message_contains_service_names(self):
        from pf_recovery_agent.agent import _build_user_message

        outage = self._make_outage(affected_services=["PFGateway", "PFOrchestrator"])
        msg = _build_user_message(outage)
        assert "PFGateway" in msg
        assert "PFOrchestrator" in msg

    def test_message_contains_symptoms(self):
        from pf_recovery_agent.agent import _build_user_message

        outage = self._make_outage(symptoms=["HTTP 502", "health probe failing"])
        msg = _build_user_message(outage)
        assert "HTTP 502" in msg
        assert "health probe failing" in msg

    def test_message_contains_regions_and_clusters(self):
        from pf_recovery_agent.agent import _build_user_message

        outage = self._make_outage(affected_regions=["eastus"], affected_clusters=["PF-Cluster-EastUS"])
        msg = _build_user_message(outage)
        assert "eastus" in msg
        assert "PF-Cluster-EastUS" in msg

    def test_message_omits_empty_optional_sections(self):
        from pf_recovery_agent.agent import _build_user_message

        outage = self._make_outage()
        msg = _build_user_message(outage)
        assert "Affected Regions" not in msg
        assert "Observed Symptoms" not in msg


class TestExtractNumberedList:
    def test_extracts_numbered_items(self):
        from pf_recovery_agent.agent import _extract_numbered_list

        text = "Some header\n1. Do thing A\n2. Do thing B\n3. Do thing C"
        items = _extract_numbered_list(text)
        assert items == ["Do thing A", "Do thing B", "Do thing C"]

    def test_handles_parenthesis_notation(self):
        from pf_recovery_agent.agent import _extract_numbered_list

        text = "1) First\n2) Second"
        items = _extract_numbered_list(text)
        assert items == ["First", "Second"]

    def test_empty_text_returns_empty_list(self):
        from pf_recovery_agent.agent import _extract_numbered_list

        assert _extract_numbered_list("") == []

    def test_ignores_non_numbered_lines(self):
        from pf_recovery_agent.agent import _extract_numbered_list

        text = "Summary:\nThis is the summary.\n\n1. Step one\n\nConclusion."
        items = _extract_numbered_list(text)
        assert items == ["Step one"]


class TestExtractSummary:
    def test_extracts_summary_section(self):
        from pf_recovery_agent.agent import _extract_summary

        text = "## Summary\n\nThe gateway is down.\n\n## Recommended Actions\n\n1. Restart pods"
        summary = _extract_summary(text)
        assert "gateway" in summary.lower()

    def test_falls_back_to_first_paragraph(self):
        from pf_recovery_agent.agent import _extract_summary

        text = "The gateway is experiencing a critical outage.\n\nMore details here."
        summary = _extract_summary(text)
        assert "gateway" in summary.lower()

    def test_empty_text_returns_empty(self):
        from pf_recovery_agent.agent import _extract_summary

        assert _extract_summary("") == ""


class TestParseResponse:
    def test_extracts_rcas_from_tool_messages(self):
        import json

        from pf_recovery_agent.agent import _parse_response

        rca_payload = json.dumps(
            [{"id": "RCA-001", "title": "Test", "root_cause": "x", "time_to_mitigate_minutes": 20}]
        )
        messages = [
            {"role": "user", "content": "outage"},
            {"role": "tool", "content": rca_payload},
        ]
        response = _parse_response("ICM-001", "Summary text", messages)
        assert len(response.relevant_rcas) == 1
        assert response.relevant_rcas[0]["id"] == "RCA-001"

    def test_extracts_tsgs_from_tool_messages(self):
        import json

        from pf_recovery_agent.agent import _parse_response

        tsg_payload = json.dumps(
            [{"id": "TSG-001", "title": "Test TSG", "diagnostic_steps": ["step1"], "mitigation_steps": []}]
        )
        messages = [{"role": "tool", "content": tsg_payload}]
        response = _parse_response("ICM-001", "", messages)
        assert len(response.relevant_tsgs) == 1
        assert response.relevant_tsgs[0]["id"] == "TSG-001"

    def test_extracts_dependency_risks(self):
        import json

        from pf_recovery_agent.agent import _parse_response

        dep_payload = json.dumps(
            {"tier1_risks": ["PFStorageService"], "blast_radius": ["PFStorageService", "PFAuditLog"]}
        )
        messages = [{"role": "tool", "content": dep_payload}]
        response = _parse_response("ICM-001", "", messages)
        assert "PFStorageService" in response.dependency_risks

    def test_estimated_rto_from_rca_data(self):
        import json

        from pf_recovery_agent.agent import _parse_response

        rca_payload = json.dumps(
            [
                {"id": "RCA-001", "root_cause": "x", "time_to_mitigate_minutes": 45},
                {"id": "RCA-002", "root_cause": "y", "time_to_mitigate_minutes": 30},
            ]
        )
        messages = [{"role": "tool", "content": rca_payload}]
        response = _parse_response("ICM-001", "", messages)
        assert response.estimated_rto_minutes == 45

    def test_deduplicates_dependency_risks(self):
        import json

        from pf_recovery_agent.agent import _parse_response

        # Same service appearing in two tool calls
        dep1 = json.dumps({"tier1_risks": ["PFGateway"], "blast_radius": ["PFGateway", "PFWorkerPool"]})
        dep2 = json.dumps({"tier1_risks": ["PFGateway"], "blast_radius": ["PFGateway"]})
        messages = [
            {"role": "tool", "content": dep1},
            {"role": "tool", "content": dep2},
        ]
        response = _parse_response("ICM-001", "", messages)
        assert response.dependency_risks.count("PFGateway") == 1
