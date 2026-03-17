"""
Unit tests for SoulBuddyClassifier.

AutoModel.from_pretrained is fully mocked — no model download required.
Tests cover: architecture (head dimensions), forward pass output shapes,
CLS token pooling, and the output tuple contract.

Skipped automatically when torch is not installed (e.g. in CI environments
that don't include the ML dependencies).
"""

import pytest
torch = pytest.importorskip("torch", reason="torch not installed — skipping ML model tests")
import torch.nn as nn
from unittest.mock import MagicMock, patch


# ============================================================================
# Helpers
# ============================================================================

HIDDEN_SIZE = 64  # small hidden size for fast tests


def _make_mock_base_model(hidden_size: int = HIDDEN_SIZE):
    """Return a mock transformer base model with the given hidden size."""
    mock_model = MagicMock(spec=nn.Module)
    mock_model.config = MagicMock()
    mock_model.config.hidden_size = hidden_size
    mock_model.parameters = MagicMock(return_value=iter([]))

    def fake_forward(input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
        output = MagicMock()
        output.last_hidden_state = last_hidden_state
        return output

    mock_model.side_effect = fake_forward
    return mock_model


def _make_classifier(hidden_size: int = HIDDEN_SIZE):
    """Instantiate SoulBuddyClassifier with a mocked base model."""
    from transformer_models.SoulBuddyClassifier import SoulBuddyClassifier

    with patch("transformer_models.SoulBuddyClassifier.AutoModel.from_pretrained",
               return_value=_make_mock_base_model(hidden_size)):
        clf = SoulBuddyClassifier("mental/mental-roberta-base")
    return clf


# ============================================================================
# Architecture tests
# ============================================================================

class TestSoulBuddyClassifierArchitecture:

    def test_situation_head_output_dim_is_8(self):
        clf = _make_classifier()
        assert clf.situation_head.out_features == 8

    def test_severity_head_output_dim_is_3(self):
        clf = _make_classifier()
        assert clf.severity_head.out_features == 3

    def test_intent_head_output_dim_is_8(self):
        clf = _make_classifier()
        assert clf.intent_head.out_features == 8

    def test_risk_head_output_dim_is_1(self):
        clf = _make_classifier()
        assert clf.risk_head.out_features == 1

    def test_all_heads_input_dim_matches_hidden_size(self):
        clf = _make_classifier(hidden_size=128)
        for head in (clf.situation_head, clf.severity_head, clf.intent_head, clf.risk_head):
            assert head.in_features == 128

    def test_is_nn_module(self):
        clf = _make_classifier()
        assert isinstance(clf, nn.Module)

    def test_base_model_is_stored(self):
        clf = _make_classifier()
        assert clf.base_model is not None


# ============================================================================
# Forward pass — output contract
# ============================================================================

class TestSoulBuddyClassifierForward:

    def _run_forward(self, batch_size: int = 2, seq_len: int = 16):
        clf = _make_classifier()
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        return clf(input_ids, attention_mask), batch_size

    def test_forward_returns_tuple_of_four(self):
        outputs, _ = self._run_forward()
        assert isinstance(outputs, tuple)
        assert len(outputs) == 4

    def test_situation_logits_shape(self):
        (situation, _, _, _), batch_size = self._run_forward(batch_size=3)
        assert situation.shape == (batch_size, 8)

    def test_severity_logits_shape(self):
        (_, severity, _, _), batch_size = self._run_forward(batch_size=3)
        assert severity.shape == (batch_size, 3)

    def test_intent_logits_shape(self):
        (_, _, intent, _), batch_size = self._run_forward(batch_size=3)
        assert intent.shape == (batch_size, 8)

    def test_risk_logits_shape(self):
        (_, _, _, risk), batch_size = self._run_forward(batch_size=3)
        assert risk.shape == (batch_size, 1)

    def test_forward_single_item_batch(self):
        (situation, severity, intent, risk), _ = self._run_forward(batch_size=1)
        assert situation.shape[0] == 1
        assert severity.shape[0] == 1
        assert intent.shape[0] == 1
        assert risk.shape[0] == 1

    def test_forward_outputs_are_tensors(self):
        outputs, _ = self._run_forward()
        for tensor in outputs:
            assert isinstance(tensor, torch.Tensor)

    def test_forward_outputs_require_no_grad_in_eval_mode(self):
        clf = _make_classifier()
        clf.eval()
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        with torch.no_grad():
            outputs = clf(input_ids, attention_mask)
        assert all(not t.requires_grad for t in outputs)


# ============================================================================
# CLS token pooling
# ============================================================================

class TestSoulBuddyClassifierCLSPooling:

    def test_cls_token_is_position_zero(self):
        """Verify that the classifier uses last_hidden_state[:, 0] (CLS token)."""
        from transformer_models.SoulBuddyClassifier import SoulBuddyClassifier

        captured = {}

        class TrackedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MagicMock()
                self.config.hidden_size = HIDDEN_SIZE

            def forward(self, input_ids, attention_mask):
                batch = input_ids.shape[0]
                seq = input_ids.shape[1]
                lhs = torch.zeros(batch, seq, HIDDEN_SIZE)
                # Set a unique value at position 0 (CLS) and something else at pos 1
                lhs[:, 0, :] = 1.0
                lhs[:, 1, :] = 99.0
                captured["lhs"] = lhs
                output = MagicMock()
                output.last_hidden_state = lhs
                return output

        tracked = TrackedModel()

        with patch("transformer_models.SoulBuddyClassifier.AutoModel.from_pretrained",
                   return_value=tracked):
            clf = SoulBuddyClassifier("mental/mental-roberta-base")

        # Replace heads with identity-like transforms to trace the pooled value
        clf.situation_head = nn.Linear(HIDDEN_SIZE, 8, bias=False)
        nn.init.ones_(clf.situation_head.weight)

        input_ids = torch.randint(0, 100, (1, 8))
        attention_mask = torch.ones(1, 8, dtype=torch.long)
        (situation, _, _, _) = clf(input_ids, attention_mask)

        # If CLS token (all 1.0) was pooled, each output neuron = sum of 1.0 * HIDDEN_SIZE
        expected_val = float(HIDDEN_SIZE)
        assert abs(situation[0, 0].item() - expected_val) < 1e-3, (
            f"Expected CLS pooling to produce {expected_val}, got {situation[0, 0].item()}"
        )
