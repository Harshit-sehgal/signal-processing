"""Tests for synthetic and semi-synthetic signal generation."""

import numpy as np

from pg_amcd.synthetic import generate_semisynthetic_chatter


def test_generate_semisynthetic_chatter_has_controlled_component():
    """Injected chatter is scaled to the requested SNR and the output is reproducible."""
    fs = 10_000.0
    rng = np.random.default_rng(0)
    stable = rng.normal(0.0, 1.0, 2000)
    t, combined, components = generate_semisynthetic_chatter(
        stable, fs, chatter_freq=1250.0, chatter_onset=0.1, snr_db=10.0, seed=42
    )
    assert t.size == stable.size
    assert combined.size == stable.size
    assert "stable" in components
    assert "chatter" in components
    assert "clean" in components
    # The clean reference equals the stable background.
    np.testing.assert_allclose(components["clean"], stable)
    # Chatter component is nonzero.
    assert np.linalg.norm(components["chatter"]) > 0.0
    # Combined equals stable + chatter.
    np.testing.assert_allclose(combined, stable + components["chatter"], atol=1e-9)


def test_generate_semisynthetic_chatter_snr_contract():
    """The injected chatter power matches the requested chatter-to-stable SNR."""
    fs = 10_000.0
    rng = np.random.default_rng(1)
    stable = rng.normal(0.0, 1.0, 4000)
    snr_db = 8.0
    _t, _combined, components = generate_semisynthetic_chatter(
        stable, fs, chatter_freq=1500.0, chatter_onset=0.2, snr_db=snr_db, seed=7
    )
    stable_power = float(np.var(components["stable"]))
    chatter_power = float(np.var(components["chatter"]))
    measured_snr_db = 10.0 * np.log10(stable_power / chatter_power)
    assert abs(measured_snr_db - snr_db) < 0.5
