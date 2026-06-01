import torch
import mpmath as mp
import pytest
from zeta_hunter.pcf.engine import PCFEngine


@pytest.fixture
def engine():
    return PCFEngine(depth=500)


def test_engine_sqrt2_minus1(engine):
    """
    CF for sqrt(2) - 1:
      a(n) = 1  (constant), b(n) = 2  (constant)
      CF = 1/(2 + 1/(2 + 1/(2 + ...))) = sqrt(2) - 1
    """
    a = torch.tensor([[1.0]], dtype=torch.float64)
    b = torch.tensor([[2.0]], dtype=torch.float64)
    result = engine.batch_pcf(a, b)
    expected = float(mp.sqrt(2) - 1)
    assert abs(result[0].item() - expected) < 1e-8


def test_engine_batch_consistency(engine):
    """Batching N copies of the same formula must give N identical results."""
    a = torch.tensor([[1.0]] * 100, dtype=torch.float64)
    b = torch.tensor([[2.0]] * 100, dtype=torch.float64)
    results = engine.batch_pcf(a, b)
    assert results.shape == (100,)
    assert (results - results[0]).abs().max().item() < 1e-12


def test_engine_no_nan(engine):
    """Random well-conditioned polynomial families should not produce NaN."""
    torch.manual_seed(42)
    a = torch.randint(-5, 6, (1000, 3)).double()
    b = torch.randint(1, 10, (1000, 3)).double()
    results = engine.batch_pcf(a, b)
    assert not torch.isnan(results).any()


def test_engine_mpmath_agrees(engine):
    """GPU float64 and mpmath must agree to 10 digits for the sqrt(2)-1 CF."""
    a_gpu = torch.tensor([[1.0]], dtype=torch.float64)
    b_gpu = torch.tensor([[2.0]], dtype=torch.float64)
    gpu_val = engine.batch_pcf(a_gpu, b_gpu)[0].item()
    mp_val = engine.precise_pcf([1], [2], dps=100)
    assert abs(gpu_val - float(mp_val)) < 1e-10


def test_engine_apery_gpu_matches_mpmath(engine):
    """
    Apery polynomials a(n) = -n^6, b(n) = 34n^3 - 51n^2 + 27n - 5.
    GPU and mpmath evaluations must agree to 8 digits at depth=500.
    (The exact relationship to zeta(3) involves an outer normalisation
    factor handled by the sweeper, not the engine.)
    """
    a_coeffs = [0, 0, 0, 0, 0, 0, -1]   # -n^6
    b_coeffs = [-5, 27, -51, 34]          # 34n^3 - 51n^2 + 27n - 5

    a_gpu = torch.tensor([a_coeffs], dtype=torch.float64)
    b_gpu = torch.tensor([b_coeffs], dtype=torch.float64)
    gpu_val = engine.batch_pcf(a_gpu, b_gpu)[0].item()

    mp_val = engine.precise_pcf(a_coeffs, b_coeffs, dps=50)
    assert mp_val is not None
    assert abs(gpu_val - float(mp_val)) < 1e-8
