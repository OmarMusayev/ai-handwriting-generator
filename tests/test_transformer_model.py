# tests/test_transformer_model.py
import torch
import torch.nn as nn
from unittest.mock import patch
from models.transformer_synthesis import TextEncoder, PositionalEncoding, StyleVAE, StrokeDecoder, MDNHead, HandWritingSynthesisTransformer


def test_positional_encoding_output_shape():
    pe = PositionalEncoding(d_model=256)
    x = torch.randn(2, 10, 256)  # (batch, seq, d_model)
    out = pe(x)
    assert out.shape == (2, 10, 256)


def test_positional_encoding_adds_signal():
    pe = PositionalEncoding(d_model=256, dropout=0.0)
    x = torch.zeros(1, 5, 256)
    out = pe(x)
    # output should not be all zeros — positional signal was added
    assert out.abs().sum() > 0


def test_text_encoder_output_shape():
    enc = TextEncoder(vocab_size=77, d_model=256, nhead=8, num_layers=4, ff_dim=512)
    text = torch.randint(0, 77, (2, 15))       # (batch=2, text_len=15)
    text_mask = torch.ones(2, 15, dtype=torch.float)
    out = enc(text, text_mask)
    assert out.shape == (2, 15, 256)


def test_text_encoder_respects_padding_mask():
    """Padded positions should not crash, shape is preserved, and valid positions are not NaN."""
    enc = TextEncoder(vocab_size=77, d_model=256, nhead=8, num_layers=4, ff_dim=512)
    enc.eval()
    text = torch.randint(0, 77, (1, 10))
    mask_half = torch.zeros(1, 10)
    mask_half[0, :5] = 1.0
    with torch.no_grad():
        out_half = enc(text, mask_half)
    assert out_half.shape == (1, 10, 256)
    assert not torch.isnan(out_half[:, :5]).any(), "valid positions should not be NaN"


# --- StyleVAE tests ---

def test_style_vae_output_shapes():
    """z, mu, logvar should all be (batch, 64)."""
    vae = StyleVAE()
    strokes = torch.randn(3, 20, 3)  # (batch=3, seq_len=20, 3)
    z, mu, logvar = vae(strokes, use_sampling=True)
    assert z.shape == (3, 64), f"z shape {z.shape} != (3, 64)"
    assert mu.shape == (3, 64), f"mu shape {mu.shape} != (3, 64)"
    assert logvar.shape == (3, 64), f"logvar shape {logvar.shape} != (3, 64)"


def test_style_vae_training_mode_uses_reparameterization():
    """With use_sampling=True, z should differ from mu (sampling adds noise) at least once in 50 runs."""
    vae = StyleVAE()
    vae.eval()  # module in eval mode; use_sampling flag still controls reparameterization
    strokes = torch.randn(2, 15, 3)
    found_difference = False
    for _ in range(50):
        with torch.no_grad():
            z, mu, logvar = vae(strokes, use_sampling=True)
        if not torch.allclose(z, mu):
            found_difference = True
            break
    assert found_difference, "Expected z != mu at least once in 50 samples (reparameterization should add noise)"


def test_style_vae_inference_mode_is_deterministic():
    """With training=False, z should equal mu exactly (no sampling noise)."""
    vae = StyleVAE()
    vae.eval()
    strokes = torch.randn(2, 15, 3)
    with torch.no_grad():
        z, mu, logvar = vae(strokes, use_sampling=False)
    assert torch.equal(z, mu), "In inference mode (use_sampling=False), z must equal mu exactly"


def test_style_vae_kl_loss_is_non_negative():
    """KL divergence = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) should be >= 0."""
    vae = StyleVAE()
    strokes = torch.randn(4, 25, 3)
    with torch.no_grad():
        z, mu, logvar = vae(strokes, use_sampling=True)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    assert kl.item() >= 0, f"KL divergence should be non-negative, got {kl.item()}"


def test_style_vae_output_is_finite_with_extreme_logvar():
    """Extreme fc_logvar weights must not produce inf/nan z."""
    vae = StyleVAE()
    with torch.no_grad():
        nn.init.constant_(vae.fc_logvar.weight, 10.0)
        nn.init.constant_(vae.fc_logvar.bias, 10.0)
    strokes = torch.randn(2, 15, 3)
    z, mu, logvar = vae(strokes, use_sampling=True)
    assert torch.isfinite(z).all(), "z must be finite even with extreme logvar"


# --- StrokeDecoder tests ---

def test_stroke_decoder_output_shape():
    """Output must be (batch, seq_len, d_model=256)."""
    decoder = StrokeDecoder()
    decoder.eval()
    batch, seq_len, text_len = 2, 10, 15
    strokes = torch.randn(batch, seq_len, 3)
    text_embeddings = torch.randn(batch, text_len, 256)
    text_padding_mask = torch.ones(batch, text_len)
    z = torch.randn(batch, 64)
    with torch.no_grad():
        out = decoder(strokes, text_embeddings, text_padding_mask, z)
    assert out.shape == (batch, seq_len, 256), f"Expected ({batch}, {seq_len}, 256), got {out.shape}"


def test_stroke_decoder_causal_mask():
    """Step 0 output must be identical when processing seq_len=1 vs seq_len=2 (causality)."""
    decoder = StrokeDecoder()
    decoder.eval()
    batch, text_len = 1, 8
    text_embeddings = torch.randn(batch, text_len, 256)
    text_padding_mask = torch.ones(batch, text_len)
    z = torch.randn(batch, 64)
    # Two strokes: only use the first to check causality
    strokes_2 = torch.randn(batch, 2, 3)
    strokes_1 = strokes_2[:, :1, :]  # first step only

    with torch.no_grad():
        out_len1 = decoder(strokes_1, text_embeddings, text_padding_mask, z)
        out_len2 = decoder(strokes_2, text_embeddings, text_padding_mask, z)

    # Step 0 output must match regardless of whether step 1 is present
    assert torch.allclose(out_len1[:, 0, :], out_len2[:, 0, :], atol=1e-5), \
        "Causal mask violated: step 0 output changed when step 1 was added"


def test_stroke_decoder_text_padding_mask():
    """Partial text padding mask must not cause crash or NaN in output."""
    decoder = StrokeDecoder()
    decoder.eval()
    batch, seq_len, text_len = 2, 5, 12
    strokes = torch.randn(batch, seq_len, 3)
    text_embeddings = torch.randn(batch, text_len, 256)
    # Only first 6 text positions are valid
    text_padding_mask = torch.zeros(batch, text_len)
    text_padding_mask[:, :6] = 1.0
    z = torch.randn(batch, 64)
    with torch.no_grad():
        out = decoder(strokes, text_embeddings, text_padding_mask, z)
    assert out.shape == (batch, seq_len, 256)
    assert torch.isfinite(out).all(), "Output must be finite with partial padding mask"


def test_stroke_decoder_past_kv_none_works():
    """Passing past_kv=None must produce the same output as omitting it entirely."""
    decoder = StrokeDecoder()
    decoder.eval()
    batch, seq_len, text_len = 2, 7, 10
    strokes = torch.randn(batch, seq_len, 3)
    text_embeddings = torch.randn(batch, text_len, 256)
    text_padding_mask = torch.ones(batch, text_len)
    z = torch.randn(batch, 64)
    with torch.no_grad():
        out_default = decoder(strokes, text_embeddings, text_padding_mask, z)
        out_none = decoder(strokes, text_embeddings, text_padding_mask, z, past_kv=None)
    assert torch.equal(out_default, out_none), \
        "past_kv=None must produce identical output to calling without past_kv"


# --- MDNHead tests ---

def test_mdn_head_output_shape():
    """MDNHead(decoder_hidden) -> (batch, seq_len, 121)."""
    mdn = MDNHead(d_model=256)
    decoder_hidden = torch.randn(3, 10, 256)  # (batch=3, seq_len=10, d_model=256)
    out = mdn(decoder_hidden)
    assert out.shape == (3, 10, 121), f"Expected (3, 10, 121), got {out.shape}"


# --- HandWritingSynthesisTransformer tests ---

def test_transformer_forward_output_shapes():
    """forward() returns (y_hat, mu, logvar) with correct shapes."""
    model = HandWritingSynthesisTransformer(vocab_size=10, d_model=256, nhead=8,
                                            text_layers=4, dec_layers=6, ff_dim=512,
                                            latent_dim=64)
    model.eval()
    batch, seq_len, text_len, style_len = 2, 15, 8, 10
    strokes = torch.randn(batch, seq_len, 3)
    text = torch.randint(0, 10, (batch, text_len))
    text_mask = torch.ones(batch, text_len)
    style_strokes = torch.randn(batch, style_len, 3)
    with torch.no_grad():
        y_hat, mu, logvar = model(strokes, text, text_mask, style_strokes, use_sampling=True)
    assert y_hat.shape == (batch, seq_len, 121), f"y_hat shape {y_hat.shape} != ({batch}, {seq_len}, 121)"
    assert mu.shape == (batch, 64), f"mu shape {mu.shape} != ({batch}, 64)"
    assert logvar.shape == (batch, 64), f"logvar shape {logvar.shape} != ({batch}, 64)"


def test_transformer_generate_returns_numpy_array():
    """generate() on a 1-sample input returns numpy ndarray with shape (1, T, 3), T > 0."""
    import numpy as np
    model = HandWritingSynthesisTransformer(vocab_size=10, d_model=256, nhead=8,
                                            text_layers=4, dec_layers=6, ff_dim=512,
                                            latent_dim=64)
    model.eval()
    text = torch.randint(0, 10, (1, 5))
    text_mask = torch.ones(1, 5)
    style_strokes = torch.randn(1, 8, 3)
    result = model.generate(text, text_mask, style_strokes, bias=1.0, max_steps=5)
    assert isinstance(result, np.ndarray), f"Expected numpy ndarray, got {type(result)}"
    assert result.ndim == 3, f"Expected 3-dim array, got {result.ndim}-dim"
    assert result.shape[0] == 1, f"Expected batch dim=1, got {result.shape[0]}"
    assert result.shape[1] > 0, "Expected at least 1 generated step"
    assert result.shape[2] == 3, f"Expected stroke dim=3, got {result.shape[2]}"


def test_transformer_generate_eos_stops_early():
    """If EOS is sampled on first step, generation stops after 1 step."""
    import numpy as np
    model = HandWritingSynthesisTransformer(vocab_size=10, d_model=256, nhead=8,
                                            text_layers=4, dec_layers=6, ff_dim=512,
                                            latent_dim=64)
    model.eval()
    text = torch.randint(0, 10, (1, 5))
    text_mask = torch.ones(1, 5)
    style_strokes = torch.randn(1, 8, 3)

    # Patch sample_from_out_dist to always return EOS=1
    eos_sample = torch.zeros(1, 1, 3)
    eos_sample[0, 0, 0] = 1.0  # eos flag = 1

    with patch("models.transformer_synthesis.sample_from_out_dist", return_value=eos_sample):
        result = model.generate(text, text_mask, style_strokes, bias=1.0, max_steps=50)

    assert isinstance(result, np.ndarray), f"Expected numpy ndarray, got {type(result)}"
    assert result.shape[1] == 1, f"Expected 1 step (EOS on first step), got {result.shape[1]}"
