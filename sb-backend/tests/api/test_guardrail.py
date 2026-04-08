"""
Unit tests for Guardrail API: POST /guardrail and shared out-of-scope behavior.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.guardrail import router as guardrail_router


@pytest.fixture
def app():
    from api.supabase_auth import verify_supabase_token
    app = FastAPI()
    app.include_router(guardrail_router, prefix="/api/v1", tags=["Guardrail"])
    # Override auth dependency so unit tests don't need a real Supabase token
    app.dependency_overrides[verify_supabase_token] = lambda: {"id": "test-user-id"}
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestGuardrailEndpointUnit:
    def test_guardrail_validates_message_required(self, client):
        resp = client.post("/api/v1/guardrail", json={})
        assert resp.status_code == 422

    def test_guardrail_empty_message_rejected(self, client):
        resp = client.post("/api/v1/guardrail", json={"message": ""})
        assert resp.status_code == 422

    def test_guardrail_flags_general_knowledge_question(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "What is the capital of France?", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["guardrail"]["is_out_of_scope"] is True
        assert data["guardrail"]["reason"] == "general_knowledge"
        assert "SoulGym" in data["guardrail"]["response"]

    def test_guardrail_flags_gibberish_as_out_of_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "blrg zqxv prst", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is True
        assert data["guardrail"]["reason"] == "nonsense"

    def test_guardrail_flags_single_token_keyboard_run_as_out_of_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "asdfghjkl", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is True
        assert data["guardrail"]["reason"] == "nonsense"

    def test_guardrail_flags_symbol_split_random_letters_as_out_of_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "fhowijvnaiewlnaces'da", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is True
        assert data["guardrail"]["reason"] == "nonsense"

    def test_guardrail_flags_symbol_noisy_consonant_token_as_out_of_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "zxcb\\", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is True
        assert data["guardrail"]["reason"] == "nonsense"

    def test_guardrail_flags_single_letter_runs_as_out_of_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "a s d f g h", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is True
        assert data["guardrail"]["reason"] == "nonsense"

    def test_guardrail_flags_mixed_alphanumeric_gibberish_as_out_of_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "infwbu94f873ucn39uq8f sad jfn9c2893fh83fh", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is True
        assert data["guardrail"]["reason"] == "nonsense"

    def test_guardrail_flags_single_mixed_alphanumeric_token_as_out_of_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "f9qu3hvleiurbvierowfeca", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is True
        assert data["guardrail"]["reason"] == "nonsense"

    @pytest.mark.parametrize(
        "message",
        [
            "jfiuven p89q3h4iofnwci9o3anfic",
            "qweoiu zxcmnv",
            "jfcn983ounwfvico4wij2039j'f[",
        ],
    )
    def test_guardrail_flags_prompt_gibberish_examples_as_out_of_scope(self, client, message):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": message, "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is True
        assert data["guardrail"]["reason"] == "nonsense"

    def test_guardrail_keeps_wellbeing_message_in_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "I've been feeling really stressed about work lately", "domain": "employee"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is False
        assert data["guardrail"]["reason"] == "in_scope"
        assert data["guardrail"]["response"] == ""

    def test_guardrail_keeps_psychology_definition_question_in_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "definition of mindfulness", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is False
        assert data["guardrail"]["reason"] == "in_scope"

    def test_guardrail_keeps_spiritual_concept_question_in_scope(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "what is shadow work in spirituality?", "domain": "general"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is False
        assert data["guardrail"]["reason"] == "in_scope"

    def test_guardrail_bad_json_falls_back_to_in_scope(self, client):
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value="not-json",
        ):
            resp = client.post(
                "/api/v1/guardrail",
                json={"message": "Can you review this merger thesis?", "domain": "corporate"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is False
        assert data["guardrail"]["reason"] == "in_scope"

    def test_guardrail_llm_failure_falls_back_to_in_scope(self, client):
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            side_effect=RuntimeError("ollama down"),
        ):
            resp = client.post(
                "/api/v1/guardrail",
                json={"message": "Can you review this merger thesis?", "domain": "corporate"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is False
        assert data["guardrail"]["reason"] == "in_scope"

    def test_guardrail_uses_domain_specific_response(self, client):
        resp = client.post(
            "/api/v1/guardrail",
            json={"message": "What is the capital of France?", "domain": "employee"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"]["is_out_of_scope"] is True
        assert "work stress" in data["guardrail"]["response"]
