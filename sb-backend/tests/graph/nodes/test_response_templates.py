"""
Unit tests for response_templates.py

Covers get_template_response priority, domain selection, and None fallback.
No external dependencies — pure unit tests.
"""

import pytest
from graph.nodes.agentic_nodes.response_templates import (
    get_template_response,
    _HIGH_RISK_TEMPLATES,
    _GREETING_TEMPLATES,
    get_out_of_scope_response,
)


# ============================================================================
# Crisis takes priority over greeting
# ============================================================================

class TestTemplatePriority:
    def test_crisis_takes_priority_over_greeting(self):
        # Both flags True — crisis must win
        result = get_template_response(is_crisis_detected=True, is_greeting=True, domain="general")
        assert result in _HIGH_RISK_TEMPLATES

    def test_crisis_detected_returns_crisis_template(self):
        result = get_template_response(is_crisis_detected=True, is_greeting=False)
        assert result in _HIGH_RISK_TEMPLATES

    def test_crisis_template_is_non_empty_string(self):
        result = get_template_response(is_crisis_detected=True, is_greeting=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_crisis_template_contains_helpline(self):
        # All crisis templates should reference at least one helpline number
        result = get_template_response(is_crisis_detected=True, is_greeting=False)
        assert any(kw in result for kw in ["9152987821", "1860-2662-345", "91-22-27546669"])


# ============================================================================
# Greeting domain routing
# ============================================================================

class TestGreetingTemplates:
    def test_greeting_student_domain(self):
        result = get_template_response(is_crisis_detected=False, is_greeting=True, domain="student")
        assert result in _GREETING_TEMPLATES["student"]

    def test_greeting_employee_domain(self):
        result = get_template_response(is_crisis_detected=False, is_greeting=True, domain="employee")
        assert result in _GREETING_TEMPLATES["employee"]

    def test_greeting_corporate_domain(self):
        result = get_template_response(is_crisis_detected=False, is_greeting=True, domain="corporate")
        assert result in _GREETING_TEMPLATES["corporate"]

    def test_greeting_general_domain(self):
        result = get_template_response(is_crisis_detected=False, is_greeting=True, domain="general")
        assert result in _GREETING_TEMPLATES["general"]

    def test_greeting_unknown_domain_falls_back_to_general(self):
        result = get_template_response(is_crisis_detected=False, is_greeting=True, domain="unknown_domain")
        assert result in _GREETING_TEMPLATES["general"]

    def test_greeting_none_domain_falls_back_to_general(self):
        result = get_template_response(is_crisis_detected=False, is_greeting=True, domain=None)
        assert result in _GREETING_TEMPLATES["general"]

    def test_greeting_template_is_non_empty_string(self):
        result = get_template_response(is_crisis_detected=False, is_greeting=True, domain="general")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_greeting_template_pool_contains_soulbuddy(self):
        # At least one template per domain should mention "SoulBuddy"
        for domain, templates in _GREETING_TEMPLATES.items():
            assert any("SoulBuddy" in t for t in templates), (
                f"Domain {domain!r}: expected at least one template to mention 'SoulBuddy'"
            )


# ============================================================================
# No template — None returned
# ============================================================================

class TestNoTemplate:
    def test_both_false_returns_none(self):
        result = get_template_response(is_crisis_detected=False, is_greeting=False)
        assert result is None

    def test_default_args_return_none(self):
        result = get_template_response(is_crisis_detected=False, is_greeting=False, domain="general")
        assert result is None


# ============================================================================
# Out-of-scope templates
# ============================================================================

class TestOutOfScopeTemplates:
    def test_out_of_scope_general_knowledge_response_mentions_support_redirect(self):
        result = get_template_response(
            is_crisis_detected=False,
            is_greeting=False,
            domain="general",
            is_out_of_scope=True,
            out_of_scope_reason="general_knowledge",
        )
        assert isinstance(result, str)
        assert "SoulGym" in result

    def test_out_of_scope_employee_response_mentions_work_stress(self):
        result = get_template_response(
            is_crisis_detected=False,
            is_greeting=False,
            domain="employee",
            is_out_of_scope=True,
            out_of_scope_reason="other_out_of_scope",
        )
        assert "work stress" in result

    def test_out_of_scope_reason_changes_response_copy(self):
        general_knowledge = get_out_of_scope_response("general", reason="general_knowledge")
        nonsense = get_out_of_scope_response("general", reason="nonsense")
        assert general_knowledge != nonsense


# ============================================================================
# Template pool completeness
# ============================================================================

class TestTemplatePoolCompleteness:
    def test_high_risk_templates_has_multiple_variants(self):
        assert len(_HIGH_RISK_TEMPLATES) >= 3

    def test_all_domains_have_multiple_greeting_variants(self):
        for domain, templates in _GREETING_TEMPLATES.items():
            assert len(templates) >= 2, f"Domain {domain!r} should have at least 2 greeting variants"

    def test_all_required_domains_present(self):
        required = {"student", "employee", "corporate", "general"}
        assert required.issubset(_GREETING_TEMPLATES.keys())
