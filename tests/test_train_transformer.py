"""Tests for TransformerHandwritingDataset and collate_fn (Task 6)."""
import os
import tempfile
import numpy as np
import torch
import pytest

from train_transformer import (
    TransformerHandwritingDataset,
    collate_fn,
    get_beta,
    compute_loss,
    train_epoch,
    validation_epoch,
    save_checkpoint,
    load_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_stroke(length: int) -> np.ndarray:
    """Create a synthetic stroke array of shape (length, 3)."""
    rng = np.random.default_rng(42)
    stroke = np.zeros((length, 3), dtype=np.float32)
    stroke[:, 0] = rng.integers(0, 2, size=length).astype(np.float32)  # eos flag
    stroke[:, 1] = rng.standard_normal(length).astype(np.float32)      # dx
    stroke[:, 2] = rng.standard_normal(length).astype(np.float32)      # dy
    return stroke


def _make_dataset(stroke_lengths=None, texts=None):
    """Build a minimal TransformerHandwritingDataset for testing."""
    if stroke_lengths is None:
        stroke_lengths = [100, 200]
    if texts is None:
        texts = [list("hello"), list("world!")]

    strokes = np.empty(len(stroke_lengths), dtype=object)
    for i, l in enumerate(stroke_lengths):
        strokes[i] = _make_stroke(l)

    texts_arr = np.empty(len(texts), dtype=object)
    for i, t in enumerate(texts):
        texts_arr[i] = np.array(t)

    # Simple char_to_id
    all_chars = set(c for t in texts for c in t)
    char_to_id = {c: idx + 1 for idx, c in enumerate(sorted(all_chars))}
    char_to_id[" "] = 0  # space / unknown → 0

    train_mean = np.array([0.0, 0.0], dtype=np.float32)
    train_std = np.array([1.0, 1.0], dtype=np.float32)

    dataset = TransformerHandwritingDataset(
        strokes=strokes,
        texts=texts_arr,
        char_to_id=char_to_id,
        train_mean=train_mean,
        train_std=train_std,
        max_stroke_len=1000,
        split_lo=0.2,
        split_hi=0.4,
    )
    return dataset


# ---------------------------------------------------------------------------
# Test 1: getitem returns correct keys and lengths sum correctly
# ---------------------------------------------------------------------------

def test_dataset_getitem_shape():
    dataset = _make_dataset(stroke_lengths=[100], texts=[list("hello")])
    item = dataset[0]

    assert set(item.keys()) == {"style_strokes", "target_strokes", "text", "text_mask"}

    style = item["style_strokes"]
    target = item["target_strokes"]
    text = item["text"]
    text_mask = item["text_mask"]

    # style and target should be numpy arrays
    assert isinstance(style, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert isinstance(text, np.ndarray)
    assert isinstance(text_mask, np.ndarray)

    # shapes
    assert style.ndim == 2 and style.shape[1] == 3
    assert target.ndim == 2 and target.shape[1] == 3

    # lengths sum to original stroke length (clamped to max_stroke_len=1000)
    total = min(100, 1000)
    assert style.shape[0] + target.shape[0] == total

    # text shape
    assert text.ndim == 1
    assert text.shape == text_mask.shape

    # text_mask values are 0 or 1
    assert set(text_mask.tolist()).issubset({0.0, 1.0})
    # all chars are valid (no padding in __getitem__)
    assert np.all(text_mask == 1.0)


# ---------------------------------------------------------------------------
# Test 2: split points are random across calls
# ---------------------------------------------------------------------------

def test_dataset_split_point_is_random():
    dataset = _make_dataset(stroke_lengths=[200], texts=[list("testing")])

    split_points = set()
    for _ in range(10):
        item = dataset[0]
        split_points.add(item["style_strokes"].shape[0])

    # With 10 independent samples and uniform random in [0.2, 0.4]*200=[40,80],
    # we expect more than one distinct split point.
    assert len(split_points) > 1, "Split points should vary across calls (randomness check)"


# ---------------------------------------------------------------------------
# Test 3: collate_fn pads to batch max lengths
# ---------------------------------------------------------------------------

def test_collate_fn_pads_to_batch_max():
    # Create two items with known, different lengths by using fixed splits
    rng = np.random.default_rng(0)

    def make_item(style_len, target_len, text_len):
        style = rng.standard_normal((style_len, 3)).astype(np.float32)
        target = rng.standard_normal((target_len, 3)).astype(np.float32)
        text = np.ones(text_len, dtype=np.int64)
        text_mask = np.ones(text_len, dtype=np.float32)
        return {
            "style_strokes": style,
            "target_strokes": target,
            "text": text,
            "text_mask": text_mask,
        }

    item1 = make_item(style_len=20, target_len=80, text_len=5)
    item2 = make_item(style_len=30, target_len=60, text_len=8)

    batch = collate_fn([item1, item2])

    # Shapes
    assert batch["style_strokes"].shape == (2, 30, 3)
    assert batch["style_mask"].shape == (2, 30)
    assert batch["target_strokes"].shape == (2, 80, 3)
    assert batch["target_mask"].shape == (2, 80)
    assert batch["target_strokes_input"].shape == (2, 80, 3)
    assert batch["text"].shape == (2, 8)
    assert batch["text_mask"].shape == (2, 8)

    # Tensor types
    assert batch["style_strokes"].dtype == torch.float32
    assert batch["target_strokes"].dtype == torch.float32
    assert batch["target_strokes_input"].dtype == torch.float32
    assert batch["text"].dtype == torch.long
    assert batch["style_mask"].dtype == torch.float32
    assert batch["target_mask"].dtype == torch.float32
    assert batch["text_mask"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 4: masks are correct (1 for valid, 0 for padding)
# ---------------------------------------------------------------------------

def test_collate_fn_masks_are_correct():
    rng = np.random.default_rng(1)

    def make_item(style_len, target_len, text_len):
        style = rng.standard_normal((style_len, 3)).astype(np.float32)
        target = rng.standard_normal((target_len, 3)).astype(np.float32)
        text = np.ones(text_len, dtype=np.int64)
        text_mask = np.ones(text_len, dtype=np.float32)
        return {
            "style_strokes": style,
            "target_strokes": target,
            "text": text,
            "text_mask": text_mask,
        }

    item1 = make_item(style_len=10, target_len=50, text_len=4)
    item2 = make_item(style_len=20, target_len=70, text_len=7)

    batch = collate_fn([item1, item2])

    # style_mask: item1 has 10 valid, 10 padded; item2 has 20 valid, 0 padded
    assert torch.all(batch["style_mask"][0, :10] == 1.0)
    assert torch.all(batch["style_mask"][0, 10:] == 0.0)
    assert torch.all(batch["style_mask"][1, :20] == 1.0)

    # target_mask: item1 has 50 valid, 20 padded; item2 has 70 valid
    assert torch.all(batch["target_mask"][0, :50] == 1.0)
    assert torch.all(batch["target_mask"][0, 50:] == 0.0)
    assert torch.all(batch["target_mask"][1, :70] == 1.0)

    # text_mask: item1 has 4 valid, 3 padded; item2 has 7 valid
    assert torch.all(batch["text_mask"][0, :4] == 1.0)
    assert torch.all(batch["text_mask"][0, 4:] == 0.0)
    assert torch.all(batch["text_mask"][1, :7] == 1.0)

    # Padding values in style_strokes should be zero
    assert torch.all(batch["style_strokes"][0, 10:] == 0.0)

    # target_strokes_input: first token for each item should be zeros (start token)
    assert torch.all(batch["target_strokes_input"][:, 0, :] == 0.0)

    # target_strokes_input[i, 1:valid_len, :] == target_strokes[i, :valid_len-1, :]
    tgt1_len = 50
    assert torch.allclose(
        batch["target_strokes_input"][0, 1:tgt1_len, :],
        batch["target_strokes"][0, :tgt1_len - 1, :],
    )


# ---------------------------------------------------------------------------
# Task 7 Tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 5: get_beta KL annealing schedule
# ---------------------------------------------------------------------------

def test_get_beta_schedule():
    assert get_beta(0) == 0.0
    assert get_beta(19) == 0.0
    assert get_beta(40) == pytest.approx(0.5)
    assert get_beta(60) == 1.0
    assert get_beta(100) == 1.0
    # Linear midpoint: epoch=20 => 0.0, epoch=60 => 1.0, epoch=40 => 0.5
    assert get_beta(20) == pytest.approx(0.0)
    assert get_beta(59) == pytest.approx(39 / 40)


# ---------------------------------------------------------------------------
# Test 6: compute_loss returns finite values
# ---------------------------------------------------------------------------

def test_compute_loss_returns_finite():
    batch_size, seq_len = 2, 5
    torch.manual_seed(0)
    y_hat = torch.randn(batch_size, seq_len, 121)
    target = torch.randn(batch_size, seq_len, 3)
    target[:, :, 0] = torch.randint(0, 2, (batch_size, seq_len)).float()  # eos is 0 or 1
    mask = torch.ones(batch_size, seq_len)
    mu = torch.randn(batch_size, 64)
    logvar = torch.zeros(batch_size, 64)
    loss, nll, kl = compute_loss(y_hat, target, mask, mu, logvar, beta=1.0)
    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)


# ---------------------------------------------------------------------------
# Test 7: train_epoch updates model weights
# ---------------------------------------------------------------------------

def _make_tiny_dataset(n_items=2, seq_len=10, text_len=5):
    """Build a minimal dataset for training tests."""
    rng = np.random.default_rng(42)

    strokes = np.empty(n_items, dtype=object)
    texts_arr = np.empty(n_items, dtype=object)
    vocab = list("abcde ")
    char_to_id = {c: i for i, c in enumerate(vocab)}

    for i in range(n_items):
        stroke = np.zeros((seq_len, 3), dtype=np.float32)
        stroke[:, 0] = rng.integers(0, 2, size=seq_len).astype(np.float32)
        stroke[:, 1:] = rng.standard_normal((seq_len, 2)).astype(np.float32)
        strokes[i] = stroke
        texts_arr[i] = np.array(list("abcde")[:text_len])

    train_mean = np.array([0.0, 0.0], dtype=np.float32)
    train_std = np.array([1.0, 1.0], dtype=np.float32)

    dataset = TransformerHandwritingDataset(
        strokes=strokes,
        texts=texts_arr,
        char_to_id=char_to_id,
        train_mean=train_mean,
        train_std=train_std,
        max_stroke_len=seq_len,
        split_lo=0.3,
        split_hi=0.5,
    )
    return dataset, char_to_id


def test_train_epoch_updates_weights():
    from models.transformer_synthesis import HandWritingSynthesisTransformer
    from torch.utils.data import DataLoader

    dataset, char_to_id = _make_tiny_dataset(n_items=2, seq_len=15)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    model = HandWritingSynthesisTransformer(vocab_size=len(char_to_id))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Snapshot weights before
    before = {k: v.clone() for k, v in model.named_parameters()}

    train_epoch(model, loader, optimizer, device=torch.device("cpu"), beta=0.0)

    # At least some parameters should have changed
    any_changed = any(
        not torch.allclose(before[k], v) for k, v in model.named_parameters()
    )
    assert any_changed, "Expected at least one parameter to change after train_epoch"


# ---------------------------------------------------------------------------
# Test 8: save/load checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_save_load_checkpoint_roundtrip():
    from models.transformer_synthesis import HandWritingSynthesisTransformer

    model = HandWritingSynthesisTransformer(vocab_size=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    state = {
        "epoch": 42,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": 3.14,
        "beta": 0.75,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "subdir", "checkpoint.pt")
        save_checkpoint(state, ckpt_path)
        assert os.path.exists(ckpt_path)

        # Create fresh model/optimizer/scheduler for loading
        model2 = HandWritingSynthesisTransformer(vocab_size=10)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5)

        epoch, best_val_loss, beta = load_checkpoint(
            ckpt_path, model2, optimizer2, scheduler2, device=torch.device("cpu")
        )

    assert epoch == 42
    assert best_val_loss == pytest.approx(3.14)
    assert beta == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Test 9: validation_epoch does not update weights
# ---------------------------------------------------------------------------

def test_validation_epoch_no_grad():
    from models.transformer_synthesis import HandWritingSynthesisTransformer
    from torch.utils.data import DataLoader

    dataset, char_to_id = _make_tiny_dataset(n_items=2, seq_len=15)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    model = HandWritingSynthesisTransformer(vocab_size=len(char_to_id))

    # Snapshot weights before
    before = {k: v.clone() for k, v in model.named_parameters()}

    val_loss = validation_epoch(model, loader, device=torch.device("cpu"), beta=0.0)

    # No parameters should have changed
    all_same = all(
        torch.allclose(before[k], v) for k, v in model.named_parameters()
    )
    assert all_same, "validation_epoch should not modify model weights"
    assert isinstance(val_loss, float)
    assert np.isfinite(val_loss)
