import mpmath as mp
import pytest
from zeta_hunter.pslq import PSLQSearcher
from zeta_hunter.constants import CONSTANTS


@pytest.fixture
def searcher():
    return PSLQSearcher(precision=100)


def test_zeta2_found(searcher):
    """zeta(2) = pi^2/6 must be found with small coefficients."""
    results = searcher.run_all_bases(CONSTANTS["zeta2"])
    verdicts = [r.verdict for r in results]
    assert "CANDIDATE" in verdicts or "IDENTITY" in verdicts
    best = max(results, key=lambda r: r.precision_digits)
    assert best.precision_digits > 50
    assert max(abs(c) for c in best.coefficients) < 1000


def test_all_bases_run_without_error(searcher):
    """run_all_bases must not raise for any target."""
    results = searcher.run_all_bases(CONSTANTS["zeta3"])
    assert isinstance(results, list)


def test_noise_classified_correctly(searcher):
    """A target with no relation to our bases should produce no IDENTITY."""
    weird_target = mp.sqrt(2) + mp.e + mp.sqrt(3) + mp.mpf("0.123456789")
    results = searcher.run_all_bases(weird_target)
    for r in results:
        assert r.verdict != "IDENTITY"


def test_result_has_formula_str(searcher):
    """Every result must include a human-readable formula_str."""
    results = searcher.run_all_bases(CONSTANTS["zeta2"])
    for r in results:
        assert isinstance(r.formula_str, str)
        assert len(r.formula_str) > 0
