"""
Task 10: Tests for Transformer adapter path in generate_conditional_sequence().
TDD — these are written before the implementation.
"""
import numpy as np
import pytest
import torch

from models.transformer_synthesis import HandWritingSynthesisTransformer
from generate import generate_conditional_sequence


# ---------------------------------------------------------------------------
# Minimal fake-transformer that passes isinstance check
# ---------------------------------------------------------------------------

class FakeTransformer(HandWritingSynthesisTransformer):
    """Subclass so isinstance(..., HandWritingSynthesisTransformer) is True."""

    def __init__(self):
        # Skip the real __init__ to avoid needing config params
        pass

    def generate(self, text, text_mask, style_strokes, bias, max_steps):
        return np.zeros((1, 5, 3))

    def eval(self):
        return self


# Minimal char_to_id for tests
_CHAR_TO_ID = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
_IDX_TO_CHAR = lambda arr: [str(x) for x in arr]  # noqa: E731


# ---------------------------------------------------------------------------
# Test 1: transformer path is used when model is a HandWritingSynthesisTransformer
# ---------------------------------------------------------------------------

def test_generate_conditional_sequence_uses_transformer_path():
    model = FakeTransformer()

    result, phi = generate_conditional_sequence(
        model_or_path=model,
        char_seq="hello",
        device="cpu",
        char_to_id=_CHAR_TO_ID,
        idx_to_char=_IDX_TO_CHAR,
        bias=1.0,
        prime=False,
        prime_seq=None,
        real_text="",
        is_map=False,
    )

    # generate() returns (1, 5, 3)
    assert result.shape == (1, 5, 3), f"Unexpected shape: {result.shape}"
    assert phi == []


# ---------------------------------------------------------------------------
# Test 2: LSTM path is used for a non-Transformer model (string path)
# We patch torch.load and HandWritingSynthesisNet so we don't need actual files.
# ---------------------------------------------------------------------------

def test_generate_conditional_sequence_uses_lstm_path(monkeypatch):
    """LSTM path is taken when model_or_path is a string (file path)."""
    from unittest.mock import MagicMock
    import generate as gen_module

    fake_lstm = MagicMock()
    fake_lstm.init_hidden.return_value = (None, None, None)
    fake_lstm.generate.return_value = np.zeros((1, 10, 3))
    fake_lstm._phi = []
    fake_lstm.to = MagicMock(return_value=fake_lstm)
    fake_lstm.eval = MagicMock(return_value=fake_lstm)
    fake_lstm.load_state_dict = MagicMock()

    # Patch HandWritingSynthesisNet in generate module's namespace
    monkeypatch.setattr(gen_module, "HandWritingSynthesisNet", lambda **kwargs: fake_lstm)
    # Patch torch.load in generate module's namespace
    monkeypatch.setattr(gen_module.torch, "load", lambda path, map_location=None: {})

    result, phi = generate_conditional_sequence(
        model_or_path="fake/path/model.pt",
        char_seq="hi",
        device="cpu",
        char_to_id=_CHAR_TO_ID,
        idx_to_char=_IDX_TO_CHAR,
        bias=1.0,
        prime=False,
        prime_seq=None,
        real_text="",
        is_map=False,
    )

    fake_lstm.generate.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3: phi is empty list for transformer path
# ---------------------------------------------------------------------------

def test_generate_transformer_phi_is_empty():
    model = FakeTransformer()

    _, phi = generate_conditional_sequence(
        model_or_path=model,
        char_seq="test",
        device="cpu",
        char_to_id=_CHAR_TO_ID,
        idx_to_char=_IDX_TO_CHAR,
        bias=2.0,
        prime=False,
        prime_seq=None,
        real_text="",
        is_map=True,  # even with is_map=True, transformer returns empty phi
    )

    assert phi == [], f"Expected empty phi, got: {phi}"


# ---------------------------------------------------------------------------
# Test 4: None prime_seq produces zero style_strokes inside generate()
# ---------------------------------------------------------------------------

def test_generate_transformer_handles_none_prime_seq():
    """When prime_seq is None, generate() is called with zeros(1,1,3) style."""

    received_style = {}

    class InspectingTransformer(HandWritingSynthesisTransformer):
        def __init__(self):
            pass

        def generate(self, text, text_mask, style_strokes, bias, max_steps):
            received_style["tensor"] = style_strokes
            return np.zeros((1, 3, 3))

        def eval(self):
            return self

    model = InspectingTransformer()

    generate_conditional_sequence(
        model_or_path=model,
        char_seq="abc",
        device="cpu",
        char_to_id=_CHAR_TO_ID,
        idx_to_char=_IDX_TO_CHAR,
        bias=1.0,
        prime=False,
        prime_seq=None,
        real_text="",
        is_map=False,
    )

    style = received_style["tensor"]
    assert isinstance(style, torch.Tensor), "style_strokes should be a Tensor"
    assert style.shape == (1, 1, 3), f"Expected (1,1,3), got {style.shape}"
    assert style.sum().item() == 0.0, "Expected all-zero style_strokes"
