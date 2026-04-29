"""Unit tests for knowledge base loader."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestKnowledgeBaseLoader:
    def test_load_rcas_returns_list(self):
        from pf_recovery_agent.knowledge_base.loader import load_rcas

        rcas = load_rcas()
        assert isinstance(rcas, list)
        assert len(rcas) > 0

    def test_load_tsgs_returns_list(self):
        from pf_recovery_agent.knowledge_base.loader import load_tsgs

        tsgs = load_tsgs()
        assert isinstance(tsgs, list)
        assert len(tsgs) > 0

    def test_load_service_dependencies_returns_list(self):
        from pf_recovery_agent.knowledge_base.loader import load_service_dependencies

        deps = load_service_dependencies()
        assert isinstance(deps, list)
        assert len(deps) > 0

    def test_rca_has_required_fields(self):
        from pf_recovery_agent.knowledge_base.loader import load_rcas

        for rca in load_rcas():
            assert rca.id
            assert rca.title
            assert isinstance(rca.affected_services, list)
            assert isinstance(rca.mitigation_steps, list)
            assert rca.time_to_mitigate_minutes > 0

    def test_tsg_has_required_fields(self):
        from pf_recovery_agent.knowledge_base.loader import load_tsgs

        for tsg in load_tsgs():
            assert tsg.id
            assert tsg.title
            assert isinstance(tsg.applicable_services, list)
            assert isinstance(tsg.diagnostic_steps, list)
            assert isinstance(tsg.mitigation_steps, list)

    def test_dependency_has_required_fields(self):
        from pf_recovery_agent.knowledge_base.loader import load_service_dependencies

        for dep in load_service_dependencies():
            assert dep.service_name
            assert dep.tier >= 1

    def test_all_rca_ids_unique(self):
        from pf_recovery_agent.knowledge_base.loader import load_rcas

        ids = [r.id for r in load_rcas()]
        assert len(ids) == len(set(ids)), "Duplicate RCA IDs found"

    def test_all_tsg_ids_unique(self):
        from pf_recovery_agent.knowledge_base.loader import load_tsgs

        ids = [t.id for t in load_tsgs()]
        assert len(ids) == len(set(ids)), "Duplicate TSG IDs found"
