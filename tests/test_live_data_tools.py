"""
Unit tests for the three live Azure data tools.

All Azure SDK calls are mocked – no live Azure credentials required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# azure_monitor_tool tests
# ---------------------------------------------------------------------------

class TestGetLiveTelemetry:
    """Tests for get_live_telemetry (Azure Monitor Log Analytics)."""

    def test_returns_error_when_not_configured(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.azure_monitor_tool.USE_AZURE_MONITOR", False)
        from pf_recovery_agent.tools.azure_monitor_tool import get_live_telemetry

        result = get_live_telemetry(["PFGateway"])
        assert "error" in result
        assert "not configured" in result["error"].lower()

    def test_returns_error_when_sdk_not_importable(self, monkeypatch):
        """When USE_AZURE_MONITOR is True but the SDK raises ImportError, error is returned gracefully."""
        monkeypatch.setattr("pf_recovery_agent.tools.azure_monitor_tool.USE_AZURE_MONITOR", True)

        import importlib
        import pf_recovery_agent.tools.azure_monitor_tool as mod

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "azure.monitor.query":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        importlib.reload(mod)

        result = mod.get_live_telemetry(["PFGateway"], workspace_id="ws-123")
        assert "error" in result

    def test_successful_query_returns_expected_keys(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.azure_monitor_tool.USE_AZURE_MONITOR", True)

        def _make_table(columns, rows):
            tbl = MagicMock()
            tbl.columns = columns
            tbl.rows = rows
            return tbl

        mock_response = MagicMock()
        mock_response.status = "Success"
        mock_response.tables = [
            _make_table(
                ["cloud_RoleName", "resultCode", "total_requests", "failed_requests", "error_rate_pct", "p99_duration_ms"],
                [["PFGateway", "502", 100, 80, 80.0, 2500]],
            )
        ]

        mock_client_instance = MagicMock()
        mock_client_instance.query_workspace.return_value = mock_response

        mock_status = MagicMock()
        mock_status.SUCCESS = "Success"
        mock_status.PARTIAL = "Partial"

        with (
            patch("azure.monitor.query.LogsQueryClient", return_value=mock_client_instance),
            patch("azure.monitor.query.LogsQueryStatus", mock_status),
            patch("pf_recovery_agent.credential.get_credential", return_value=MagicMock()),
        ):
            # Re-import inside patch context so lazy imports pick up the mock
            import importlib
            import pf_recovery_agent.tools.azure_monitor_tool as mod
            importlib.reload(mod)

            result = mod.get_live_telemetry(["PFGateway"], workspace_id="ws-123")

        assert "failed_requests" in result
        assert "top_exceptions" in result
        assert "failed_dependencies" in result
        assert "availability_issues" in result
        assert "summary" in result
        assert result["services_analysed"] == ["PFGateway"]

    def test_exception_during_query_adds_warnings(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.azure_monitor_tool.USE_AZURE_MONITOR", True)

        mock_client_instance = MagicMock()
        mock_client_instance.query_workspace.side_effect = Exception("connection refused")

        mock_status = MagicMock()
        mock_status.SUCCESS = "Success"
        mock_status.PARTIAL = "Partial"

        with (
            patch("azure.monitor.query.LogsQueryClient", return_value=mock_client_instance),
            patch("azure.monitor.query.LogsQueryStatus", mock_status),
            patch("pf_recovery_agent.credential.get_credential", return_value=MagicMock()),
        ):
            import importlib
            import pf_recovery_agent.tools.azure_monitor_tool as mod
            importlib.reload(mod)

            result = mod.get_live_telemetry(["PFGateway"], workspace_id="ws-123")

        assert "warnings" in result

    def test_service_filter_clause_empty(self):
        from pf_recovery_agent.tools.azure_monitor_tool import _service_filter_clause

        assert _service_filter_clause([]) == ""

    def test_service_filter_clause_single(self):
        from pf_recovery_agent.tools.azure_monitor_tool import _service_filter_clause

        clause = _service_filter_clause(["PFGateway"])
        assert "PFGateway" in clause
        assert "cloud_RoleName" in clause

    def test_service_filter_clause_multiple(self):
        from pf_recovery_agent.tools.azure_monitor_tool import _service_filter_clause

        clause = _service_filter_clause(["PFGateway", "PFOrchestrator"])
        assert "PFGateway" in clause
        assert "PFOrchestrator" in clause

    def test_rows_to_dicts_converts_correctly(self):
        from pf_recovery_agent.tools.azure_monitor_tool import _rows_to_dicts

        table = MagicMock()
        table.columns = ["a", "b", "c"]
        table.rows = [[1, 2, 3], [4, 5, 6]]
        result = _rows_to_dicts(table)
        assert result == [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]

    def test_rows_to_dicts_none_table(self):
        from pf_recovery_agent.tools.azure_monitor_tool import _rows_to_dicts

        assert _rows_to_dicts(None) == []


# ---------------------------------------------------------------------------
# resource_health_tool tests
# ---------------------------------------------------------------------------

class TestGetResourceHealth:
    def test_returns_error_when_not_configured(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.resource_health_tool.USE_AZURE_RESOURCE_HEALTH", False)
        from pf_recovery_agent.tools.resource_health_tool import get_resource_health

        result = get_resource_health()
        assert "error" in result

    def _make_availability_status(self, availability_state: str, resource_id: str):
        status = MagicMock()
        status.id = resource_id
        status.properties.availability_state = availability_state
        status.properties.summary = "Test summary"
        status.properties.reason_type = ""
        status.properties.occurred_time = None
        status.properties.recommended_actions = []
        return status

    def test_healthy_resources_classified_correctly(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.resource_health_tool.USE_AZURE_RESOURCE_HEALTH", True)

        rid = "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.ContainerService/managedClusters/pf-aks"
        mock_client_instance = MagicMock()
        mock_client_instance.availability_statuses.list_by_resource_group.return_value = [
            self._make_availability_status("Available", rid)
        ]

        with (
            patch("azure.mgmt.resourcehealth.ResourceHealthMgmtClient", return_value=mock_client_instance),
            patch("pf_recovery_agent.credential.get_credential", return_value=MagicMock()),
        ):
            import importlib
            import pf_recovery_agent.tools.resource_health_tool as mod
            importlib.reload(mod)

            result = mod.get_resource_health(subscription_id="sub-123", resource_group="rg")

        assert result["healthy_count"] == 1
        assert result["degraded"] == []
        assert "All monitored" in result["summary"]

    def test_degraded_resources_flagged(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.resource_health_tool.USE_AZURE_RESOURCE_HEALTH", True)

        rid = "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.DocumentDB/databaseAccounts/pf-cosmos"
        mock_client_instance = MagicMock()
        mock_client_instance.availability_statuses.list_by_resource_group.return_value = [
            self._make_availability_status("Degraded", rid)
        ]

        with (
            patch("azure.mgmt.resourcehealth.ResourceHealthMgmtClient", return_value=mock_client_instance),
            patch("pf_recovery_agent.credential.get_credential", return_value=MagicMock()),
        ):
            import importlib
            import pf_recovery_agent.tools.resource_health_tool as mod
            importlib.reload(mod)

            result = mod.get_resource_health(subscription_id="sub-123", resource_group="rg")

        assert len(result["degraded"]) == 1
        assert result["degraded"][0]["availability_state"] == "Degraded"
        assert "DEGRADED" in result["summary"]

    def test_extract_resource_type(self):
        from pf_recovery_agent.tools.resource_health_tool import _extract_resource_type

        rid = "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.DocumentDB/databaseAccounts/mydb"
        assert _extract_resource_type(rid) == "Microsoft.DocumentDB/databaseAccounts"

    def test_extract_resource_name(self):
        from pf_recovery_agent.tools.resource_health_tool import _extract_resource_name

        rid = "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.DocumentDB/databaseAccounts/mydb"
        assert _extract_resource_name(rid) == "mydb"

    def test_api_exception_returns_error(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.resource_health_tool.USE_AZURE_RESOURCE_HEALTH", True)

        mock_client_instance = MagicMock()
        mock_client_instance.availability_statuses.list_by_resource_group.side_effect = Exception("API error")

        with (
            patch("azure.mgmt.resourcehealth.ResourceHealthMgmtClient", return_value=mock_client_instance),
            patch("pf_recovery_agent.credential.get_credential", return_value=MagicMock()),
        ):
            import importlib
            import pf_recovery_agent.tools.resource_health_tool as mod
            importlib.reload(mod)

            result = mod.get_resource_health(subscription_id="sub-123", resource_group="rg")

        assert "error" in result


# ---------------------------------------------------------------------------
# service_health_tool tests
# ---------------------------------------------------------------------------

class TestGetAzureServiceHealth:
    def test_returns_error_when_not_configured(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.service_health_tool.USE_AZURE_RESOURCE_HEALTH", False)
        from pf_recovery_agent.tools.service_health_tool import get_azure_service_health

        result = get_azure_service_health()
        assert "error" in result

    def _make_event(self, event_type="ServiceIssue", status="Active", service="azure kubernetes service", region="East US"):
        event = MagicMock()
        event.event_type = event_type
        event.level = "Critical"
        event.title = f"Test {event_type}"
        event.summary = "Something is broken"
        event.status = status
        event.last_update_time = None
        event.impact_start_time = None
        event.tracking_id = "TRK-001"

        impact = MagicMock()
        impact.impacted_service = service
        r = MagicMock()
        r.impacted_region = region
        impact.impacted_regions = [r]
        event.impact = [impact]
        return event

    def test_active_outage_detected(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.service_health_tool.USE_AZURE_RESOURCE_HEALTH", True)

        mock_client_instance = MagicMock()
        mock_client_instance.events.list_by_subscription_id.return_value = [
            self._make_event(event_type="ServiceIssue", status="Active", service="azure kubernetes service")
        ]

        with (
            patch("azure.mgmt.resourcehealth.ResourceHealthMgmtClient", return_value=mock_client_instance),
            patch("pf_recovery_agent.credential.get_credential", return_value=MagicMock()),
        ):
            import importlib
            import pf_recovery_agent.tools.service_health_tool as mod
            importlib.reload(mod)

            result = mod.get_azure_service_health(subscription_id="sub-123")

        assert len(result["active_outages"]) == 1
        assert result["pf_impacted"] is True
        assert "ACTIVE AZURE OUTAGE" in result["summary"]

    def test_region_filter_excludes_non_matching(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.service_health_tool.USE_AZURE_RESOURCE_HEALTH", True)

        mock_client_instance = MagicMock()
        mock_client_instance.events.list_by_subscription_id.return_value = [
            self._make_event(region="West Europe")  # not eastus
        ]

        with (
            patch("azure.mgmt.resourcehealth.ResourceHealthMgmtClient", return_value=mock_client_instance),
            patch("pf_recovery_agent.credential.get_credential", return_value=MagicMock()),
        ):
            import importlib
            import pf_recovery_agent.tools.service_health_tool as mod
            importlib.reload(mod)

            result = mod.get_azure_service_health(regions=["eastus"], subscription_id="sub-123")

        assert result["active_outages"] == []
        assert result["pf_impacted"] is False

    def test_no_events_returns_clean_response(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.service_health_tool.USE_AZURE_RESOURCE_HEALTH", True)

        mock_client_instance = MagicMock()
        mock_client_instance.events.list_by_subscription_id.return_value = []

        with (
            patch("azure.mgmt.resourcehealth.ResourceHealthMgmtClient", return_value=mock_client_instance),
            patch("pf_recovery_agent.credential.get_credential", return_value=MagicMock()),
        ):
            import importlib
            import pf_recovery_agent.tools.service_health_tool as mod
            importlib.reload(mod)

            result = mod.get_azure_service_health(subscription_id="sub-123")

        assert result["active_outages"] == []
        assert result["pf_impacted"] is False
        assert "No active" in result["summary"]

    def test_api_exception_returns_error(self, monkeypatch):
        monkeypatch.setattr("pf_recovery_agent.tools.service_health_tool.USE_AZURE_RESOURCE_HEALTH", True)

        mock_client_instance = MagicMock()
        mock_client_instance.events.list_by_subscription_id.side_effect = Exception("Service unavailable")

        with (
            patch("azure.mgmt.resourcehealth.ResourceHealthMgmtClient", return_value=mock_client_instance),
            patch("pf_recovery_agent.credential.get_credential", return_value=MagicMock()),
        ):
            import importlib
            import pf_recovery_agent.tools.service_health_tool as mod
            importlib.reload(mod)

            result = mod.get_azure_service_health(subscription_id="sub-123")

        assert "error" in result
