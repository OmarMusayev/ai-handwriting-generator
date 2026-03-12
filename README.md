# ✦ HAND MAGIC ✦

> Retro-synth handwriting synthesis web app powered by deep learning.
> Made by **Omar Musayev**

Generate realistic handwriting from any text using deep learning. Draw your own style on the canvas and prime the model to mimic it.

---

## Features

- **No login required** — session persists via a long-lived cookie
- **Multi-style management** — save up to 10 hand-drawn styles, rename or delete them
- **Async generation** — samples stream progressively as each one is ready
- **Retro synthwave UI** — neon glows, CRT scanlines, perspective grid
- **FastAPI + Uvicorn** backend
- **Two model backends** — classic LSTM or Transformer + Style VAE (switchable via env var)
- **RAM-efficient singletons** — model, vocab, and normalization stats loaded once at startup

---

## Quick Start (local)

```bash
git clone https://github.com/your-handle/hand-magic.git
cd hand-magic
pip install -e ".[dev]"
cp .env.example .env
hand_gen_env/bin/python -m uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

To use the Transformer model instead of LSTM:

```bash
MODEL_TYPE=transformer MODEL_PATH=checkpoints/transformer/checkpoint_best.pt \
  hand_gen_env/bin/python -m uvicorn main:app --reload --port 8000
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DISK_STORAGE_PATH` | `./disk` | Where session/style data is stored |
| `MODEL_PATH` | `results/synthesis/best_model_synthesis_4.pt` | Path to model checkpoint |
| `MODEL_TYPE` | `lstm` | Model backend: `lstm` or `transformer` |
| `DATA_PATH` | `data/` | Path to `strokes.npy` + `sentences.txt` |
| `MAX_GEN_STEPS` | `600` | Max generation steps |
| `SESSION_TTL_DAYS` | `7` | Days before session cleanup |
| `MAX_STYLES_PER_SESSION` | `10` | Styles per session |
| `N_SAMPLES` | `5` | Samples generated per request |

---

## Models

### LSTM (production-ready)

3-layer LSTM with a soft attention window conditioned on text, predicting bivariate Gaussian mixture offsets. Architecture from [Alex Graves (2013)](https://arxiv.org/abs/1308.0850).

Checkpoint: `results/synthesis/best_model_synthesis_4.pt`

### Transformer + Style VAE (in training)

A modern architecture built on top of the LSTM baseline:

- **TextEncoder** — 4-layer Transformer encoder, d_model=256, 8 heads
- **StyleVAE** — BiLSTM (2 layers, hidden=256) → 64-dim latent space via reparameterization trick
- **StrokeDecoder** — 6-layer causal Transformer decoder with cross-attention to text and style injection at every step
- **MDNHead** — Linear(256→121), same Gaussian mixture layout as the LSTM (1 EOS + 6×20 components)

Training uses KL annealing (β-VAE): β=0 for epochs 0–19, linear ramp 0→1 for epochs 20–60, β=1 after. Loss: `(NLL + β·KL) / n`.

Checkpoint: `checkpoints/transformer/checkpoint_best.pt`

---

## Training

### Hardware

**Cloud (Transformer model — current):**
- **Training machine:** RunPod H100 SXM 80GB
- **GPU:** NVIDIA H100 SXM, 80GB HBM3, 700W TDP
- **Training speed:** ~2.7 it/s at batch_size=256, max_stroke_len=1000 (~20 min/epoch)
- **300 epochs:** ~4 hours total

**Local (LSTM model / prototyping):**
- **Training machine:** Apple MacBook Pro M4 Max, 48GB unified RAM
- **GPU:** Apple MPS (Metal Performance Shaders — Apple Silicon integrated GPU)
- **Training speed:** ~2–3 it/s at batch_size=8, max_stroke_len=1000 (~31 min/epoch)

### Data

- `data/strokes.npy` — IAM On-Line Handwriting stroke arrays, format `[eos_flag, dx, dy]`
- `data/sentences.txt` — Corresponding text transcriptions
- 6,000 samples, median length 627 steps, p95=945

### Train the Transformer model

```bash
# Fresh training
hand_gen_env/bin/python train_transformer.py \
  --epochs 300 --batch_size 8 --max_stroke_len 1000 \
  --checkpoint_dir checkpoints/transformer/ --tqdm

# Resume from latest checkpoint
hand_gen_env/bin/python train_transformer.py \
  --epochs 300 --batch_size 8 --max_stroke_len 1000 \
  --checkpoint_dir checkpoints/transformer/ --tqdm --resume
```

Important: when resuming after changing `--epochs`, the LR scheduler is reset to anneal over the remaining epochs automatically.

### Train the LSTM model

```bash
hand_gen_env/bin/python train.py
```

### Training notes

- MPS memory usage: Activity Monitor is the authoritative gauge — PyTorch's `torch.mps.current_allocated_memory()` only tracks PyTorch allocations, not MPS driver buffers
- Green memory pressure = fine, yellow = monitor swap, red = reduce batch size
- `batch_size=8` with `max_stroke_len=1000` is stable on M4 Max 48GB
- Checkpoints saved every epoch to `checkpoint_latest.pt`; best val loss saved to `checkpoint_best.pt`
- Training stats (mean/std) are embedded in the checkpoint so the app always uses the exact normalization the model was trained with

---

## Running Tests

```bash
hand_gen_env/bin/python -m pytest -v
```

---

## Project Structure

```
hand-magic/
├── main.py                        # FastAPI app entry point
├── train.py                       # LSTM training script
├── train_transformer.py           # Transformer + Style VAE training script
├── generate.py                    # CLI generation (both model types)
├── app/
│   ├── api/                       # Route handlers (styles, generate, jobs)
│   ├── core/                      # Config, singletons, session management
│   ├── services/                  # Generation worker, job store, cleanup
│   ├── static/                    # CSS + JS
│   └── templates/                 # Jinja2 HTML
├── models/
│   ├── models.py                  # HandWritingSynthesisNet (LSTM)
│   └── transformer_synthesis.py   # HandWritingSynthesisTransformer
├── utils/                         # Dataset, normalization, plotting helpers
├── checkpoints/transformer/       # Transformer training checkpoints
├── results/synthesis/             # LSTM training checkpoints
└── data/                          # Stroke data
```

---

## License

MIT — see [LICENSE](LICENSE).
