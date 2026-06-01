import mpmath as mp
import pytest
from zeta_hunter.constants import CONSTANTS, verify_identities, get_targets


def test_constants_loaded():
    assert len(CONSTANTS) >= 30
    assert "zeta3" in CONSTANTS
    assert "zeta5" in CONSTANTS
    assert "zeta7" in CONSTANTS
    assert "catalan" in CONSTANTS
    assert "4_over_pi" in CONSTANTS


def test_precision():
    pi_str = mp.nstr(CONSTANTS["pi"], 50, strip_zeros=False)
    assert pi_str.startswith("3.14159265358979323846")


def test_verify_identities_passes():
    assert verify_identities(min_digits=100) is True


def test_known_identity_zeta2():
    diff = abs(CONSTANTS["zeta2"] - mp.pi ** 2 / 6)
    assert diff < mp.mpf(10) ** -100


def test_known_identity_li3_neg1():
    # Li_3(-1) = -3/4 * zeta(3)
    diff = abs(CONSTANTS["Li3_neg1"] - (-mp.mpf(3) / 4 * CONSTANTS["zeta3"]))
    assert diff < mp.mpf(10) ** -100


def test_known_identity_beta3():
    # beta(3) = pi^3 / 32
    diff = abs(CONSTANTS["beta3"] - mp.pi ** 3 / 32)
    assert diff < mp.mpf(10) ** -100


def test_get_targets_returns_subset():
    targets = get_targets()
    for key in ("zeta3", "zeta5", "zeta7", "catalan", "4_over_pi"):
        assert key in targets
