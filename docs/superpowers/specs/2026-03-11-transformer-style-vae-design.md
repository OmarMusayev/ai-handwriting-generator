# Transformer + Style VAE Handwriting Synthesis

**Date:** 2026-03-11
**Status:** Approved
**Train target:** M4 Max MacBook (48GB unified RAM, MPS GPU)
**Inference target:** Dell Latitude 5420 (i7-1185G7, CPU-only, 16GB RAM)

---

## Problem

The current model (Graves 2013 LSTM + Gaussian Window Attention + MDN) has three concrete quality issues:

1. **Shaky strokes** — LSTM struggles with long-range stroke dependencies
2. **Text fidelity** — hand-designed Gaussian window attention misses characters, especially on longer strings
3. **Style drift** — style priming via hidden state copy fades after the first few strokes

---

## Solution

Replace the LSTM backbone with a Transformer encoder-decoder architecture, and replace hidden-state style priming with a proper VAE-based style encoder. Keep the MDN output head (with Cholesky sampling bug fixed).

---

## Architecture

### Components

```
[Text]           → Text Encoder      → text_embeddings  (seq_len × 256)
[Canvas strokes] → Style VAE Encoder → z                (64-dim vector)
[z + text_emb]   → Stroke Decoder    → MDN outputs      (121-dim per step)
[MDN outputs]    → Sample            → (eos, dx, dy)
```

### Text Encoder
- Transformer encoder, 4 layers
- d_model=256, nhead=8, ff_dim=512
- Learned character embeddings (replaces one-hot + Gaussian window)
- Output: sequence of context vectors (text_len × 256) for decoder cross-attention

### Style VAE Encoder
- Bidirectional LSTM, 2 layers, hidden=256
- Final hidden state → two linear projections → μ (64-dim), log σ (64-dim)
- Training: sample z ~ N(μ, σ) via reparameterization trick
- Inference: use μ directly (deterministic, no randomness)
- Input: a held-out portion of the same writer's strokes (different from the target)

### Stroke Decoder
- Autoregressive Transformer decoder, 6 layers
- d_model=256, nhead=8, ff_dim=512
- Causal self-attention on previous strokes
- Cross-attention to text_embeddings at every layer
- Style vector z added to stroke input embeddings at every step (not just first)
- KV cache at inference: each step is O(1) key/value lookups

### MDN Head
- Linear(256 → 121)
- Same output split: 1 EOS + 6×20 mixture params (weights, μ₁, μ₂, σ₁, σ₂, ρ)
- **Cholesky fix applied**: sampling uses L (Cholesky factor) not Σ directly
- 20 mixture components

**Total parameters:** ~12M (FP32), ~3MB (INT8 quantized)

---

## Data Flow

### Training
```
For each batch:
  style_strokes   → Style VAE → z ~ N(μ, σ)     [held-out portion of writer's strokes]
  text            → Text Encoder → text_embeddings
  strokes[0:T-1]  → Stroke Decoder (teacher forcing, cross-attn to text, conditioned on z)
  strokes[1:T]    → MDN loss targets

Loss = NLL(MDN) + β · KL(N(μ,σ) || N(0,I))
```

### Inference
```
1. User draws on canvas → style_strokes
2. Style VAE → z = μ (deterministic)
3. User types text → Text Encoder → text_embeddings
4. Decoder autoregressively generates (eos, dx, dy) with KV cache
5. Stop at eos=1 or 600 steps
6. Denormalize → plot_stroke → PNG
```

### Style Interpolation
Given two style latents z1, z2:
```
z = (1 - α) · z1 + α · z2
```
Valid because VAE regularizes latent space to be smooth (N(0,I) prior).

---

## Training Schedule

| Stage | Epochs | β (KL weight) | Notes |
|---|---|---|---|
| 1 | 0–20 | 0.0 | Pure reconstruction, decoder stabilizes |
| 2 | 20–60 | 0.0 → 1.0 (linear) | KL annealing |
| 3 | 60–100 | 1.0 | LR decay, full VAE |

- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **LR schedule:** Cosine annealing over 100 epochs
- **Gradient clipping:** norm clipping at 1.0 (not value clipping)
- **Device:** MPS (`torch.device("mps")` on M4 Max)
- **Estimated training time:** ~3-5 hours total on M4 Max MPS

---

## Checkpointing

Saves every epoch to `checkpoints/transformer/checkpoint_latest.pt`:
```python
{
  "epoch": int,
  "model_state": state_dict,
  "optimizer_state": state_dict,
  "scheduler_state": state_dict,
  "best_val_loss": float,
  "beta": float,          # current KL annealing value
}
```

Separately saves `checkpoints/transformer/checkpoint_best.pt` when validation loss improves. Resume loads `checkpoint_latest.pt` and starts from `epoch + 1`.

---

## Inference Optimization (Laptop Server)

**INT8 Dynamic Quantization** (applied at deployment, no retraining):
```python
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
)
```

**Expected inference time on i7-1185G7:**

| | Current LSTM (FP32) | New Transformer (INT8) |
|---|---|---|
| Model weights | 36MB | ~3MB |
| 600 steps, 1 sample | ~15-20s | ~6-9s |
| 600 steps, 3 samples | ~45-60s | ~18-27s |

**PyTorch thread config (unchanged):**
```python
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
```

---

## Codebase Integration

### New files
```
models/transformer_synthesis.py   — TextEncoder, StyleVAE, StrokeDecoder, MDNHead,
                                     HandWritingSynthesisTransformer (wraps all four)
train_transformer.py              — training script with checkpointing
```

### Unchanged files
```
app/                              — zero changes
utils/                            — data pipeline, plot_stroke, MDN loss all reused
generate.py                       — thin adapter so new model fits existing call signature
```

### Swapping in the app
```bash
MODEL_PATH=checkpoints/transformer/checkpoint_best.pt
MODEL_TYPE=transformer
```
`ModelSingleton` loads whichever type `MODEL_TYPE` specifies. All API routes, job runner, and session logic untouched.

---

## Quality Improvements Over Current Model

| Issue | Current (LSTM) | New (Transformer + VAE) |
|---|---|---|
| Shaky strokes | LSTM gradient issues on long seqs | Transformer self-attention, full context |
| Text fidelity | Gaussian window misses chars | Learned cross-attention, unconstrained |
| Style drift | Hidden state fades after ~50 steps | z injected at every decoder layer |
| Style interpolation | Not possible | z1 → z2 linear interpolation |

---

## Out of Scope

- Diffusion-based generation (too slow for CPU inference target)
- Hierarchical word/stroke decomposition (overengineered for demo)
- Full app rewrite or frontend changes
