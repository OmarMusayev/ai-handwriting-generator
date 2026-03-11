# tests/test_cholesky_sampling.py
import torch
import pytest
from models.models import sample_from_out_dist, sample_batch_from_out_dist


def _make_y_hat(mu1=0.0, mu2=0.0, std1=1.0, std2=1.0, rho=0.8):
    """Build a 121-dim y_hat with one dominant mixture component."""
    y = torch.zeros(121)
    y[1] = 10.0        # large logit for component 0 (dominant)
    y[21] = mu1        # mu_1[0]
    y[41] = mu2        # mu_2[0]
    y[61] = torch.log(torch.tensor(std1))
    y[81] = torch.log(torch.tensor(std2))
    y[101] = torch.atanh(torch.tensor(rho))
    return y


def test_sample_covariance_is_not_sigma_squared():
    """With high correlation, var(dx) must be ~1.0, not ~1.81."""
    torch.manual_seed(42)
    y_hat = _make_y_hat(std1=1.0, std2=1.0, rho=0.9)
    samples = [sample_from_out_dist(y_hat, bias=0.0)[0, 0, 1:] for _ in range(2000)]
    samples = torch.stack(samples)
    var_dx = samples[:, 0].var().item()
    assert var_dx < 1.5, (
        f"var(dx)={var_dx:.3f} — likely still using Σ instead of Cholesky L"
    )


def test_sample_returns_correct_shape():
    torch.manual_seed(0)
    y_hat = _make_y_hat()
    out = sample_from_out_dist(y_hat, bias=0.0)
    assert out.shape == (1, 1, 3)


def test_eos_is_zero_or_one():
    torch.manual_seed(0)
    y_hat = _make_y_hat()
    for _ in range(20):
        out = sample_from_out_dist(y_hat, bias=0.0)
        assert out[0, 0, 0].item() in (0.0, 1.0)


def test_sample_batch_batch_size_1_no_crash():
    """batch_size=1 must not crash due to over-squeezing."""
    torch.manual_seed(0)
    y_hat = torch.zeros(1, 121)
    y_hat[0, 1] = 10.0   # dominant component 0
    y_hat[0, 61] = 0.0   # logstd1 = 0 → std=1
    y_hat[0, 81] = 0.0   # logstd2 = 0 → std=1
    out = sample_batch_from_out_dist(y_hat, bias=0.0)
    assert out.shape == (1, 1, 3)
