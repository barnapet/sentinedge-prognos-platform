import json
import math

import numpy as np
import pandas as pd
import pytest

from src.features import build_dataset
from src.features.build_dataset import build_experiment, main
from src.features.extraction import (
    EXPERIMENTS,
    FEATURE_COLUMNS,
    add_rolling_rms_ratio,
    add_rolling_skewness,
    compute_kurtosis,
    compute_rms,
    compute_skewness,
    extract_experiment_features,
)
from src.features.candidate_features import (
    CREST_FACTOR_COLUMNS,
    FREQUENCY_DOMAIN_COLUMNS,
    SAMPLING_RATE_HZ,
    add_rolling_crest_factor,
    compute_amplitude_spectrum,
    compute_band_amplitude,
    compute_band_amplitude_normalised,
    compute_bearing_frequencies,
    compute_crest_factor,
    compute_envelope_spectrum,
    compute_spectral_kurtosis,
    extract_crest_factor,
    extract_frequency_domain_features,
)
from src.features.versioning import (
    build_manifest,
    compute_code_hash,
    compute_combined_hash,
    compute_raw_dataset_version,
)


def write_snapshot(path, columns):
    """Write a tab-separated snapshot file with the given per-column value lists."""
    n_rows = len(columns[0])
    with open(path, "w") as f:
        for row in range(n_rows):
            f.write("\t".join(str(col[row]) for col in columns) + "\n")


# --- compute_rms / compute_kurtosis -------------------------------------------

def test_compute_rms_matches_manual_calculation():
    signal = np.array([3.0, 4.0])
    assert compute_rms(signal) == pytest.approx(math.sqrt((9.0 + 16.0) / 2))


def test_compute_kurtosis_uses_standard_not_fisher_convention():
    """fisher=False: a symmetric two-point-mass signal has standard kurtosis 1.0,
    not scipy's default Fisher (excess) form -- Gaussian is 3, not 0, in this
    convention, matching docs/feature_windowing_decision.md's absolute-threshold
    rationale (baseline kurtosis ~3.4, not ~0)."""
    signal = np.array([-1.0, -1.0, 1.0, 1.0])
    assert compute_kurtosis(signal) == pytest.approx(1.0)


def test_compute_skewness_uses_scipy_default_biased_convention():
    """Must match `skew(sig)` in 03_feature_candidate_screening.ipynb -- scipy's
    default `bias=True` estimator, not the bias-corrected (`bias=False`) one. A
    silent switch to the corrected estimator would shift values enough to matter
    given how close to zero baseline skewness already sits (docs/eda_findings.md)."""
    from scipy.stats import skew as _reference_skew

    signal = np.array([1.0, 2.0, 2.0, 2.0, 10.0])
    assert compute_skewness(signal) == pytest.approx(_reference_skew(signal))
    assert compute_skewness(signal) != pytest.approx(_reference_skew(signal, bias=False))


# --- add_rolling_rms_ratio ------------------------------------------------------

def test_add_rolling_rms_ratio_uses_10_file_window_and_baseline():
    """Same computation as notebooks/02_health_state_labeling.ipynb's load_stats:
    10-file rolling mean of rms, min_periods=1, divided by the mean rms of the first
    50 files (docs/feature_windowing_decision.md, Issue #40)."""
    rms_values = [1.0, 2.0, 3.0] + [1.0] * 20  # first 20 rows form the baseline window
    df = pd.DataFrame({"rms": rms_values})

    result = add_rolling_rms_ratio(df, rolling_window=10, baseline_n_files=20)

    baseline = pd.Series(rms_values).head(20).mean()
    expected_rolling = pd.Series(rms_values).rolling(10, min_periods=1).mean()
    expected_ratio = expected_rolling / baseline

    assert result["rms_ratio"].tolist() == pytest.approx(expected_ratio.tolist())
    # min_periods=1 means no NaNs anywhere, including the first (window - 1) rows.
    assert not result["rms_ratio"].isna().any()


# --- add_rolling_skewness --------------------------------------------------------

def test_add_rolling_skewness_uses_10_file_window_with_no_baseline_ratio():
    """Same window as add_rolling_rms_ratio (Issue #40), but -- unlike rms_ratio --
    no division by a baseline: docs/feature_windowing_decision.md found baseline
    |skewness| ~ 0 in all three experiments, so a baseline ratio would be meaningless
    (Issue #23, docs/skewness_crestfactor_decision.md)."""
    skew_values = [0.1, -0.2, 0.3] + [0.0] * 20
    df = pd.DataFrame({"skewness": skew_values})

    result = add_rolling_skewness(df, rolling_window=10)

    expected = pd.Series(skew_values).rolling(10, min_periods=1).mean()
    assert result["skewness_smoothed"].tolist() == pytest.approx(expected.tolist())
    assert not result["skewness_smoothed"].isna().any()


# --- extract_experiment_features (end-to-end on a synthetic raw dir) -----------

def test_extract_experiment_features_end_to_end(tmp_path):
    raw_dir = tmp_path / "synthetic_test"
    raw_dir.mkdir()

    # Three tiny 4-point, single-channel snapshots, named as real IMS timestamps are.
    filenames = ["2003.10.22.12.00.00", "2003.10.22.12.10.00", "2003.10.22.12.20.00"]
    signals = [[1.0, -1.0, 1.0, -1.0], [2.0, -2.0, 2.0, -2.0], [0.5, 0.5, -0.5, -0.5]]
    for name, sig in zip(filenames, signals):
        write_snapshot(raw_dir / name, [sig])

    df = extract_experiment_features(
        raw_dir, experiment="synthetic_test", channel_idx=0, rolling_window=2, baseline_n_files=1
    )

    assert list(df.columns) == FEATURE_COLUMNS
    assert df["experiment"].tolist() == ["synthetic_test"] * 3
    assert df["file_index"].tolist() == [0, 1, 2]
    # Filenames sort chronologically, so row order must match the listed order above.
    assert df["rms"].tolist() == pytest.approx(
        [compute_rms(np.array(sig)) for sig in signals]
    )
    assert df["kurtosis"].tolist() == pytest.approx(
        [compute_kurtosis(np.array(sig)) for sig in signals]
    )
    assert df["skewness"].tolist() == pytest.approx(
        [compute_skewness(np.array(sig)) for sig in signals]
    )
    assert not df["rms_ratio"].isna().any()
    assert not df["skewness_smoothed"].isna().any()


@pytest.mark.parametrize("experiment_name", ["1st_test", "2nd_test", "3rd_test"])
def test_extract_experiment_features_reads_documented_tracked_channel(tmp_path, experiment_name):
    """Each experiment's tracked-bearing channel (docs/eda_findings.md Section 1) must
    be the column actually read -- not an arbitrary/default one. Build an 8-column
    synthetic file where every column has a distinct, known signal, and confirm the
    extracted rms/kurtosis/skewness match the documented channel_idx's column only."""
    cfg = EXPERIMENTS[experiment_name]
    raw_dir = tmp_path / experiment_name
    raw_dir.mkdir()

    n_channels = 8
    # Column j gets values [j, -j, j, -j] (well-defined, non-constant kurtosis, and
    # distinguishable between columns) so a wrong channel_idx would be caught.
    columns = [[j, -j, j, -j] for j in range(1, n_channels + 1)]
    write_snapshot(raw_dir / "2003.10.22.12.00.00", columns)

    df = extract_experiment_features(raw_dir, experiment=experiment_name, channel_idx=cfg.channel_idx)

    expected_signal = np.array(columns[cfg.channel_idx], dtype=np.float32)
    assert df["rms"].iloc[0] == pytest.approx(compute_rms(expected_signal))
    assert df["kurtosis"].iloc[0] == pytest.approx(compute_kurtosis(expected_signal))
    assert df["skewness"].iloc[0] == pytest.approx(compute_skewness(expected_signal))
    assert df["experiment"].iloc[0] == experiment_name


def test_experiment_column_allows_unambiguous_grouping_after_concat(tmp_path):
    """Issue #43 AC 2: the three experiments' outputs, once concatenated (as a
    downstream `pd.concat` over the written parquet files would do), must be
    filterable/groupable by `experiment` alone -- without relying on filename or
    row order to tell the test sets apart."""
    frames = []
    for i, experiment_name in enumerate(EXPERIMENTS):
        raw_dir = tmp_path / experiment_name
        raw_dir.mkdir()
        write_snapshot(raw_dir / "2003.10.22.12.00.00", [[float(i), -float(i)]])
        frames.append(
            extract_experiment_features(raw_dir, experiment=experiment_name, channel_idx=0)
        )

    combined = pd.concat(frames, ignore_index=True)

    assert set(combined["experiment"]) == set(EXPERIMENTS)
    for experiment_name in EXPERIMENTS:
        subset = combined[combined["experiment"] == experiment_name]
        assert len(subset) == 1


# --- candidate_features: crest factor (Issue #23, evaluated but not used) -------

def test_compute_crest_factor_matches_manual_calculation():
    signal = np.array([1.0, -2.0, 3.0, -4.0])
    expected_rms = compute_rms(signal)
    assert compute_crest_factor(signal) == pytest.approx(4.0 / expected_rms)


def test_compute_crest_factor_is_nan_for_all_zero_signal():
    """RMS == 0 would otherwise divide by zero -- docs/eda_findings.md never
    documents an all-zero snapshot, but the end-of-life rig-shutdown artifact
    (Section 2) gets close (0.0015g/0.0040g), so this must not raise."""
    signal = np.zeros(4)
    assert math.isnan(compute_crest_factor(signal))


def test_add_rolling_crest_factor_uses_configurable_window():
    """Not used by extract_crest_factor's default output (Issue #23 kept crest factor
    unwindowed, per docs/skewness_crestfactor_decision.md), but kept as a function
    rather than deleted -- this confirms it still computes correctly."""
    values = [10.0, 8.0, 6.0] + [4.0] * 20
    df = pd.DataFrame({"crest_factor": values})

    result = add_rolling_crest_factor(df, rolling_window=10)

    expected = pd.Series(values).rolling(10, min_periods=1).mean()
    assert result["crest_factor_smoothed"].tolist() == pytest.approx(expected.tolist())


def test_extract_crest_factor_end_to_end(tmp_path):
    raw_dir = tmp_path / "synthetic_test"
    raw_dir.mkdir()

    filenames = ["2003.10.22.12.00.00", "2003.10.22.12.10.00", "2003.10.22.12.20.00"]
    signals = [[1.0, -1.0, 1.0, -1.0], [2.0, -2.0, 2.0, -2.0], [0.5, 0.5, -0.5, -0.5]]
    for name, sig in zip(filenames, signals):
        write_snapshot(raw_dir / name, [sig])

    df = extract_crest_factor(raw_dir, experiment="synthetic_test", channel_idx=0)

    assert list(df.columns) == CREST_FACTOR_COLUMNS
    assert df["experiment"].tolist() == ["synthetic_test"] * 3
    assert df["file_index"].tolist() == [0, 1, 2]
    assert df["crest_factor"].tolist() == pytest.approx(
        [compute_crest_factor(np.array(sig)) for sig in signals]
    )


@pytest.mark.parametrize("experiment_name", ["1st_test", "2nd_test", "3rd_test"])
def test_extract_crest_factor_reads_documented_tracked_channel(tmp_path, experiment_name):
    cfg = EXPERIMENTS[experiment_name]
    raw_dir = tmp_path / experiment_name
    raw_dir.mkdir()

    n_channels = 8
    columns = [[j, -j, j, -j] for j in range(1, n_channels + 1)]
    write_snapshot(raw_dir / "2003.10.22.12.00.00", columns)

    df = extract_crest_factor(raw_dir, experiment=experiment_name, channel_idx=cfg.channel_idx)

    expected_signal = np.array(columns[cfg.channel_idx], dtype=np.float32)
    assert df["crest_factor"].iloc[0] == pytest.approx(compute_crest_factor(expected_signal))


# --- candidate_features: frequency domain (Issue #22, evaluated but not used) ---

def synthetic_tone(freq_hz, n=4096, fs=SAMPLING_RATE_HZ, amplitude=1.0, noise=0.0, seed=0):
    """A sine at `freq_hz`, optionally with additive Gaussian noise."""
    t = np.arange(n) / fs
    sig = amplitude * np.sin(2 * np.pi * freq_hz * t)
    if noise:
        sig = sig + np.random.default_rng(seed).normal(0, noise, n)
    return sig


def test_compute_bearing_frequencies_matches_published_values_for_this_dataset():
    """The bearing geometry is sourced from external literature, not the dataset's own
    README (docs/frequency_domain_decision.md Section 1). This test is the cross-check
    that justifies trusting it: the standard BPFO/BPFI formulas, fed that geometry and
    the *documented* 2000 RPM shaft speed, must reproduce the independently published
    characteristic frequencies for this dataset (~236.4 / ~296.9 Hz). If someone edits
    a geometry constant, this fails."""
    f = compute_bearing_frequencies()

    assert f.shaft_hz == pytest.approx(2000.0 / 60.0)
    assert f.bpfo_hz == pytest.approx(236.4, abs=0.1)
    assert f.bpfi_hz == pytest.approx(296.9, abs=0.1)
    # BPFI is always the higher of the two for a given geometry (the (1 + ratio) term).
    assert f.bpfi_hz > f.bpfo_hz


def test_compute_bearing_frequencies_scales_linearly_with_shaft_speed():
    single = compute_bearing_frequencies(shaft_rpm=1000.0)
    double = compute_bearing_frequencies(shaft_rpm=2000.0)

    assert double.bpfo_hz == pytest.approx(2 * single.bpfo_hz)
    assert double.bpfi_hz == pytest.approx(2 * single.bpfi_hz)


def test_compute_bearing_frequencies_uses_only_the_diameter_ratio():
    """Pitch/roller diameter enter the formula only as a ratio, so the unit they are
    expressed in must not matter -- inches vs. mm must give identical frequencies."""
    inches = compute_bearing_frequencies(pitch_diameter=2.815, roller_diameter=0.331)
    mm = compute_bearing_frequencies(pitch_diameter=2.815 * 25.4, roller_diameter=0.331 * 25.4)

    assert inches.bpfo_hz == pytest.approx(mm.bpfo_hz)
    assert inches.bpfi_hz == pytest.approx(mm.bpfi_hz)


def test_compute_amplitude_spectrum_locates_a_known_tone():
    freqs, spectrum = compute_amplitude_spectrum(synthetic_tone(1000.0), SAMPLING_RATE_HZ)

    assert freqs[int(np.argmax(spectrum))] == pytest.approx(1000.0, abs=10.0)
    assert len(freqs) == len(spectrum)


def test_compute_band_amplitude_is_selective_to_the_requested_frequency():
    """A band centred on the tone must pick it up; a band 60 Hz away (roughly the real
    BPFO/BPFI separation) must not -- otherwise the two defect features would be
    measuring the same thing."""
    freqs, spectrum = compute_amplitude_spectrum(synthetic_tone(236.4), SAMPLING_RATE_HZ)

    on_tone = compute_band_amplitude(freqs, spectrum, 236.4, n_harmonics=1)
    off_tone = compute_band_amplitude(freqs, spectrum, 296.9, n_harmonics=1)

    assert on_tone > 100 * off_tone


def test_compute_band_amplitude_sums_harmonics():
    """A signal with energy at f and 2f must read higher with n_harmonics=2 than with
    n_harmonics=1 -- confirming harmonics are actually summed, not ignored."""
    sig = synthetic_tone(300.0) + synthetic_tone(600.0)
    freqs, spectrum = compute_amplitude_spectrum(sig, SAMPLING_RATE_HZ)

    fundamental_only = compute_band_amplitude(freqs, spectrum, 300.0, n_harmonics=1)
    with_harmonic = compute_band_amplitude(freqs, spectrum, 300.0, n_harmonics=2)

    assert with_harmonic > 1.5 * fundamental_only


def test_compute_band_amplitude_skips_harmonics_beyond_nyquist():
    """Harmonics past Nyquist must be skipped, not wrapped around into a real bin."""
    freqs, spectrum = compute_amplitude_spectrum(synthetic_tone(100.0), SAMPLING_RATE_HZ)
    near_nyquist = SAMPLING_RATE_HZ / 2 * 0.9

    # Second harmonic of near_nyquist is beyond Nyquist; asking for 3 must not raise
    # and must equal asking for 1.
    assert compute_band_amplitude(
        freqs, spectrum, near_nyquist, n_harmonics=3
    ) == pytest.approx(compute_band_amplitude(freqs, spectrum, near_nyquist, n_harmonics=1))


def test_compute_band_amplitude_normalised_rejects_a_pure_gain_change():
    """The whole point of the normalised form: scaling the signal amplitude scales the
    raw band amplitude but must leave the noise-floor-relative one unchanged. This is
    what makes it a spectral-shape feature rather than a restatement of RMS
    (docs/frequency_domain_decision.md)."""
    sig = synthetic_tone(236.4, noise=0.1)
    freqs_1x, spec_1x = compute_amplitude_spectrum(sig, SAMPLING_RATE_HZ)
    freqs_5x, spec_5x = compute_amplitude_spectrum(5.0 * sig, SAMPLING_RATE_HZ)

    raw_1x = compute_band_amplitude(freqs_1x, spec_1x, 236.4)
    raw_5x = compute_band_amplitude(freqs_5x, spec_5x, 236.4)
    norm_1x = compute_band_amplitude_normalised(freqs_1x, spec_1x, 236.4)
    norm_5x = compute_band_amplitude_normalised(freqs_5x, spec_5x, 236.4)

    assert raw_5x == pytest.approx(5.0 * raw_1x, rel=1e-6)
    assert norm_5x == pytest.approx(norm_1x, rel=1e-6)


def test_compute_band_amplitude_normalised_is_nan_for_an_all_zero_signal():
    freqs, spectrum = compute_amplitude_spectrum(np.zeros(4096), SAMPLING_RATE_HZ)
    assert math.isnan(compute_band_amplitude_normalised(freqs, spectrum, 236.4))


def modulated_defect_signal(defect_hz, carrier_hz=4000.0, n=20480, fs=SAMPLING_RATE_HZ, seed=0):
    """A physically realistic bearing defect: a high-frequency resonance rung once per
    defect period, so the defect rate appears only as the *modulation* of the carrier
    and there is no energy at `defect_hz` itself."""
    sig = np.random.default_rng(seed).normal(0, 0.05, n)
    period = int(fs / defect_hz)
    for start in range(0, n, period):
        length = min(300, n - start)
        decay = np.exp(-np.arange(length) / 40)
        sig[start : start + length] += 1.5 * decay * np.sin(
            2 * np.pi * carrier_hz * np.arange(length) / fs
        )
    return sig


def test_compute_envelope_spectrum_recovers_a_modulated_defect_the_plain_fft_understates():
    """The reason envelope analysis is here at all: for a resonance-modulated defect
    (the physically realistic case), demodulating must surface the defect rate far more
    strongly than reading the raw spectrum at that frequency. Without this, a null
    result from the plain FFT alone would not be evidence of absence."""
    bf = compute_bearing_frequencies()
    sig = modulated_defect_signal(bf.bpfo_hz)

    plain_f, plain_s = compute_amplitude_spectrum(sig, SAMPLING_RATE_HZ)
    env_f, env_s = compute_envelope_spectrum(sig, SAMPLING_RATE_HZ)

    plain = compute_band_amplitude_normalised(plain_f, plain_s, bf.bpfo_hz)
    envelope = compute_band_amplitude_normalised(env_f, env_s, bf.bpfo_hz)

    assert envelope > 5 * plain


def test_compute_envelope_spectrum_stays_selective_between_bpfo_and_bpfi():
    bf = compute_bearing_frequencies()
    env_f, env_s = compute_envelope_spectrum(modulated_defect_signal(bf.bpfo_hz), SAMPLING_RATE_HZ)

    on_defect = compute_band_amplitude_normalised(env_f, env_s, bf.bpfo_hz)
    off_defect = compute_band_amplitude_normalised(env_f, env_s, bf.bpfi_hz)

    assert on_defect > 10 * off_defect


def test_compute_envelope_spectrum_rejects_an_out_of_range_highpass():
    with pytest.raises(ValueError):
        compute_envelope_spectrum(np.zeros(4096), SAMPLING_RATE_HZ, highpass_hz=SAMPLING_RATE_HZ)


def test_compute_spectral_kurtosis_is_near_zero_for_stationary_gaussian_noise():
    """SK is defined with a -2 offset so a stationary complex-Gaussian process sits at
    ~0. Broadband stationary noise should therefore not look impulsive."""
    noise = np.random.default_rng(0).normal(0, 1.0, 20480)
    assert compute_spectral_kurtosis(noise, SAMPLING_RATE_HZ) < 5.0


def test_compute_spectral_kurtosis_rises_for_an_impulsive_signal():
    """The property that makes SK worth testing at all: bursty energy must read higher
    than the same broadband noise without bursts."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, 20480)
    impulsive = noise.copy()
    for start in range(0, len(impulsive), 2000):
        impulsive[start : start + 20] += 3.0 * np.hanning(20)

    assert compute_spectral_kurtosis(impulsive, SAMPLING_RATE_HZ) > compute_spectral_kurtosis(
        noise, SAMPLING_RATE_HZ
    )


def test_extract_frequency_domain_features_end_to_end(tmp_path):
    raw_dir = tmp_path / "synthetic_test"
    raw_dir.mkdir()

    bf = compute_bearing_frequencies()
    filenames = ["2003.10.22.12.00.00", "2003.10.22.12.10.00"]
    # One snapshot with a BPFO tone, one without -- the BPFO column must tell them apart.
    signals = [synthetic_tone(bf.bpfo_hz, n=4096, noise=0.05), synthetic_tone(0.0, n=4096, noise=0.05)]
    for name, sig in zip(filenames, signals):
        write_snapshot(raw_dir / name, [sig.tolist()])

    df = extract_frequency_domain_features(raw_dir, experiment="synthetic_test", channel_idx=0)

    assert list(df.columns) == FREQUENCY_DOMAIN_COLUMNS
    assert df["experiment"].tolist() == ["synthetic_test"] * 2
    assert df["file_index"].tolist() == [0, 1]
    assert df["bpfo_amplitude_norm"].iloc[0] > df["bpfo_amplitude_norm"].iloc[1]


def test_extract_frequency_domain_features_reads_documented_tracked_channel(tmp_path):
    """Same guard as the time-domain extractors: the tracked bearing's channel must be
    the one actually analysed. Put a BPFO tone in only the tracked channel and flat
    noise elsewhere -- reading the wrong column would miss the tone."""
    experiment_name = "2nd_test"
    cfg = EXPERIMENTS[experiment_name]
    bf = compute_bearing_frequencies()

    raw_dir = tmp_path / experiment_name
    raw_dir.mkdir()

    n_channels = 4
    rng = np.random.default_rng(0)
    columns = [rng.normal(0, 0.05, 4096).tolist() for _ in range(n_channels)]
    columns[cfg.channel_idx] = synthetic_tone(bf.bpfo_hz, n=4096, noise=0.05).tolist()
    write_snapshot(raw_dir / "2004.02.12.10.32.39", columns)

    tracked = extract_frequency_domain_features(
        raw_dir, experiment=experiment_name, channel_idx=cfg.channel_idx
    )
    other_idx = (cfg.channel_idx + 1) % n_channels
    untracked = extract_frequency_domain_features(
        raw_dir, experiment=experiment_name, channel_idx=other_idx
    )

    assert tracked["bpfo_amplitude_norm"].iloc[0] > 10 * untracked["bpfo_amplitude_norm"].iloc[0]


# --- versioning: code hash -------------------------------------------------------

def test_compute_code_hash_is_deterministic(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")

    assert compute_code_hash((f,)) == compute_code_hash((f,))


def test_compute_code_hash_changes_with_content(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")
    original = compute_code_hash((f,))

    f.write_text("x = 2\n")
    changed = compute_code_hash((f,))

    assert original != changed


def test_compute_code_hash_independent_of_argument_order(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("a = 1\n")
    b.write_text("b = 2\n")

    assert compute_code_hash((a, b)) == compute_code_hash((b, a))


# --- versioning: raw dataset version ---------------------------------------------

def test_compute_raw_dataset_version_is_deterministic(tmp_path):
    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    (raw_dir / "2003.10.22.12.00.00").write_text("data\n")
    (raw_dir / "2003.10.22.12.10.00").write_text("more data\n")

    assert compute_raw_dataset_version(raw_dir) == compute_raw_dataset_version(raw_dir)


def test_compute_raw_dataset_version_changes_with_file_size(tmp_path):
    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    f = raw_dir / "2003.10.22.12.00.00"
    f.write_text("data\n")
    original = compute_raw_dataset_version(raw_dir)

    f.write_text("more data than before\n")
    changed = compute_raw_dataset_version(raw_dir)

    assert original != changed


def test_compute_raw_dataset_version_changes_with_file_count(tmp_path):
    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    (raw_dir / "2003.10.22.12.00.00").write_text("data\n")
    original = compute_raw_dataset_version(raw_dir)

    (raw_dir / "2003.10.22.12.10.00").write_text("data\n")
    changed = compute_raw_dataset_version(raw_dir)

    assert original != changed


# --- versioning: combined hash and manifest --------------------------------------

def test_compute_combined_hash_deterministic_and_sensitive_to_either_input():
    h1 = compute_combined_hash("code-a", "data-a")
    assert h1 == compute_combined_hash("code-a", "data-a")
    assert h1 != compute_combined_hash("code-b", "data-a")
    assert h1 != compute_combined_hash("code-a", "data-b")


def test_build_manifest_contains_required_fields():
    manifest = build_manifest(
        experiment="1st_test",
        code_hash="deadbeef",
        raw_dataset_version="cafef00d",
        feature_columns=FEATURE_COLUMNS,
        n_files=2156,
    )

    assert manifest["experiment"] == "1st_test"
    assert manifest["code_hash"] == "deadbeef"
    assert manifest["raw_dataset_version"] == "cafef00d"
    assert manifest["combined_hash"] == compute_combined_hash("deadbeef", "cafef00d")
    assert manifest["feature_columns"] == FEATURE_COLUMNS
    assert manifest["n_files"] == 2156
    assert "generated_at" in manifest


# --- reproducibility: same code + same raw data -> same hash ---------------------

def test_reproducibility_same_code_and_data_yield_same_combined_hash(tmp_path):
    code_file = tmp_path / "extraction_copy.py"
    code_file.write_text("# pretend generating code\n")

    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    (raw_dir / "2003.10.22.12.00.00").write_text("0.1\t0.2\n")
    (raw_dir / "2003.10.22.12.10.00").write_text("0.3\t0.4\n")

    def run():
        code_hash = compute_code_hash((code_file,))
        raw_version = compute_raw_dataset_version(raw_dir)
        manifest = build_manifest(
            experiment="1st_test",
            code_hash=code_hash,
            raw_dataset_version=raw_version,
            feature_columns=FEATURE_COLUMNS,
            n_files=2,
        )
        return manifest["combined_hash"]

    assert run() == run()


def test_reproducibility_breaks_if_raw_data_changes(tmp_path):
    code_file = tmp_path / "extraction_copy.py"
    code_file.write_text("# pretend generating code\n")

    raw_dir = tmp_path / "1st_test"
    raw_dir.mkdir()
    (raw_dir / "2003.10.22.12.00.00").write_text("0.1\t0.2\n")

    code_hash = compute_code_hash((code_file,))
    before = compute_combined_hash(code_hash, compute_raw_dataset_version(raw_dir))

    (raw_dir / "2003.10.22.12.10.00").write_text("0.3\t0.4\n")  # dataset changed
    after = compute_combined_hash(code_hash, compute_raw_dataset_version(raw_dir))

    assert before != after


# --- build_dataset: build_experiment / main (Issue #55) --------------------------
#
# build_experiment() is what actually produces M3's input: it ties extraction and
# versioning together and writes both artifacts to disk. Before Issue #55 it was only
# exercised indirectly, through the CI execution of
# notebooks/04_feature_pipeline_validation.ipynb -- i.e. only against the real 6.2 GB
# dataset. These tests cover it directly, on synthetic snapshots, using the same
# write_snapshot() fixture pattern as the extraction tests above.

def write_experiment_raw_dir(raw_root, experiment, n_files=3, n_channels=8):
    """Create `<raw_root>/<experiment>/` with `n_files` synthetic snapshots.

    Mirrors the real layout build_experiment() expects: it is handed the *parent*
    directory and appends the experiment name itself. Every channel carries a
    distinct, non-constant signal, so reading the wrong column is detectable.
    """
    raw_dir = raw_root / experiment
    raw_dir.mkdir(parents=True)
    for i in range(n_files):
        columns = [[float(j + i), -float(j + i), float(j), -float(j)] for j in range(1, n_channels + 1)]
        write_snapshot(raw_dir / f"2003.10.22.12.{i:02d}.00", columns)
    return raw_dir


def test_build_experiment_writes_parquet_and_manifest_to_expected_paths(tmp_path):
    """Both artifacts, at the exact documented filenames
    (docs/feature_extraction_versioning.md Section 3), and the output directory is
    created if it does not exist yet."""
    raw_root = tmp_path / "raw"
    write_experiment_raw_dir(raw_root, "2nd_test")
    processed_dir = tmp_path / "processed"  # deliberately not pre-created

    returned = build_experiment("2nd_test", raw_dir=raw_root, processed_dir=processed_dir)

    parquet_path = processed_dir / "2nd_test_features.parquet"
    manifest_path = processed_dir / "2nd_test_features_manifest.json"
    assert returned == parquet_path
    assert parquet_path.is_file()
    assert manifest_path.is_file()


@pytest.mark.parametrize("experiment_name", ["1st_test", "2nd_test", "3rd_test"])
def test_build_experiment_parquet_matches_direct_extraction(tmp_path, experiment_name):
    """The written parquet must be exactly what extract_experiment_features() returns
    for the same input -- build_experiment() adds no transformation, reordering, or
    dtype drift on the way to disk. Run per experiment, so the documented tracked
    channel (EXPERIMENTS[...].channel_idx, docs/eda_findings.md Section 1) is also
    confirmed to be wired through rather than defaulted."""
    raw_root = tmp_path / "raw"
    raw_dir = write_experiment_raw_dir(raw_root, experiment_name)

    parquet_path = build_experiment(
        experiment_name, raw_dir=raw_root, processed_dir=tmp_path / "processed"
    )

    written = pd.read_parquet(parquet_path)
    expected = extract_experiment_features(
        raw_dir, experiment=experiment_name, channel_idx=EXPERIMENTS[experiment_name].channel_idx
    )
    pd.testing.assert_frame_equal(written, expected)
    assert list(written.columns) == FEATURE_COLUMNS


def test_build_experiment_manifest_fields_match_the_written_parquet(tmp_path):
    """Every manifest field is populated from the run that produced the accompanying
    parquet -- not stubbed, and not describing a different input. n_files in
    particular is the integrity cross-check docs/feature_extraction_versioning.md
    Section 4 relies on, so it must equal the actual row count on disk."""
    raw_root = tmp_path / "raw"
    raw_dir = write_experiment_raw_dir(raw_root, "3rd_test", n_files=5)

    parquet_path = build_experiment(
        "3rd_test", raw_dir=raw_root, processed_dir=tmp_path / "processed"
    )
    manifest = json.loads((parquet_path.parent / "3rd_test_features_manifest.json").read_text())
    written = pd.read_parquet(parquet_path)

    assert manifest["experiment"] == "3rd_test"
    assert manifest["code_hash"] == compute_code_hash()
    assert manifest["raw_dataset_version"] == compute_raw_dataset_version(raw_dir)
    assert manifest["combined_hash"] == compute_combined_hash(
        manifest["code_hash"], manifest["raw_dataset_version"]
    )
    assert manifest["feature_columns"] == FEATURE_COLUMNS == list(written.columns)
    assert manifest["n_files"] == len(written) == 5
    # UTC ISO-8601, per the manifest schema -- parseable and timezone-aware, not a bare
    # local-time string a consumer would have to guess the offset for.
    generated_at = pd.Timestamp(manifest["generated_at"])
    assert generated_at.tz is not None


def test_build_experiment_is_reproducible_for_unchanged_code_and_raw_data(tmp_path):
    """The reproducibility promise in src/features/build_dataset.py's docstring, at the
    level that actually makes it: re-running against unchanged code and unchanged raw
    data reproduces the same combined_hash *and* the same parquet content. Written to
    two separate output directories so the second run cannot be reading the first's
    artifacts."""
    raw_root = tmp_path / "raw"
    write_experiment_raw_dir(raw_root, "1st_test")

    first = build_experiment("1st_test", raw_dir=raw_root, processed_dir=tmp_path / "run_a")
    second = build_experiment("1st_test", raw_dir=raw_root, processed_dir=tmp_path / "run_b")

    manifest_a = json.loads((first.parent / "1st_test_features_manifest.json").read_text())
    manifest_b = json.loads((second.parent / "1st_test_features_manifest.json").read_text())

    assert manifest_a["combined_hash"] == manifest_b["combined_hash"]
    assert manifest_a["code_hash"] == manifest_b["code_hash"]
    assert manifest_a["raw_dataset_version"] == manifest_b["raw_dataset_version"]
    pd.testing.assert_frame_equal(pd.read_parquet(first), pd.read_parquet(second))


def test_build_experiment_combined_hash_changes_when_raw_data_changes(tmp_path):
    """Counterpart to the reproducibility test above: without this, a combined_hash that
    ignored its inputs entirely would still pass. Adding a snapshot must change the
    hash, which is the whole point of the stale-cache detection
    (docs/PRD.md Section 12, Issue #41)."""
    raw_root = tmp_path / "raw"
    raw_dir = write_experiment_raw_dir(raw_root, "2nd_test")

    before = json.loads(
        (
            build_experiment("2nd_test", raw_dir=raw_root, processed_dir=tmp_path / "before").parent
            / "2nd_test_features_manifest.json"
        ).read_text()
    )

    write_snapshot(raw_dir / "2003.10.22.13.00.00", [[1.0, -1.0, 1.0, -1.0]] * 8)

    after = json.loads(
        (
            build_experiment("2nd_test", raw_dir=raw_root, processed_dir=tmp_path / "after").parent
            / "2nd_test_features_manifest.json"
        ).read_text()
    )

    assert before["combined_hash"] != after["combined_hash"]
    assert after["n_files"] == before["n_files"] + 1


def test_main_builds_every_experiment_in_the_registry(monkeypatch, capsys):
    """main() must cover all three experiments, with no silent omission.

    build_experiment itself is substituted rather than redirected: its raw_dir /
    processed_dir defaults are bound to the repo-absolute RAW_DIR / PROCESSED_DIR at
    import time, and main() calls it without arguments, so monkeypatching those module
    globals would not reach the frozen defaults. Substituting the function is what keeps
    this test off the real data/raw/ tree.
    """
    calls = []

    def fake_build_experiment(name):
        calls.append(name)
        return build_dataset.PROCESSED_DIR / f"{name}_features.parquet"

    monkeypatch.setattr(build_dataset, "build_experiment", fake_build_experiment)

    main()

    assert calls == list(EXPERIMENTS)
    printed = capsys.readouterr().out
    for name in EXPERIMENTS:
        assert name in printed
