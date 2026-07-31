"""Candidate features: computed and evaluated, but not part of the core pipeline.

This module holds feature computations that were implemented and measured against real
data, then **not** promoted into `src/features/extraction.py`'s `FEATURE_COLUMNS`. They
are kept (not deleted) so a future issue can re-evaluate them against a different
feature set, failure mode, or model -- per Issue #23's and Issue #22's explicit
instruction not to throw away evaluated-but-unused computation code.

**Crest factor (Issue #23 -- evaluated, rejected).** `docs/eda_findings.md` Section 4
flagged it "plausible, low priority" and already suspected redundancy with kurtosis
(`corr` 0.57-0.88 in the Degrading+Critical window). Issue #23 confirmed this: crest
factor correlates 0.56-0.88 with kurtosis across all three experiments, and where that
correlation is *not* high (`2nd_test`, `3rd_test`), its own univariate separability
across health states is far weaker than kurtosis's. See
`docs/skewness_crestfactor_decision.md`.

**Frequency-domain features (Issue #22 -- evaluated, rejected).** BPFO/BPFI defect-band
amplitudes (raw and noise-floor-normalized) and spectral kurtosis, the first
frequency-domain analysis performed on this dataset. See
`docs/frequency_domain_decision.md` for the bearing-geometry sourcing, the computed
characteristic frequencies, and the separability/redundancy results behind the drop
decision.

(Skewness was evaluated alongside crest factor in Issue #23 but *confirmed* useful --
it lives in `src/features/extraction.py` instead, alongside RMS/kurtosis.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.extraction import (
    ROLLING_WINDOW,
    compute_rms,
    list_snapshot_files,
    load_channel,
    parse_timestamp,
)

CREST_FACTOR_COLUMNS = ["experiment", "file_index", "timestamp", "crest_factor"]

# --- Bearing geometry and acquisition constants (Issue #22) -----------------------
#
# SOURCING, because this matters for whether BPFO/BPFI mean anything at all:
#
# Documented *in this project / in the dataset's own README PDF*:
#   - shaft speed 2000 RPM, held constant (data/raw/"Readme Document for IMS Bearing
#     Data.pdf", "Test Rig Setup"; restated in docs/PRD.md Section 6)
#   - bearing model "Rexnord ZA-2115 double row" (same PDF)
#   - sampling: 20,480 points per file at 20 kHz (same PDF; validated in data/README.md)
#
# NOT documented in the dataset PDF or anywhere in this repo -- sourced externally:
#   - roller count, pitch diameter, roller diameter, contact angle.
# These come from the published literature on this specific dataset (the geometry is a
# property of the ZA-2115, not of any one paper). They are used here because they can be
# *cross-validated*: substituting them into the standard BPFO/BPFI formulas below
# reproduces the independently published characteristic frequencies for this dataset
# (~236 Hz BPFO / ~297 Hz BPFI) to within rounding. That agreement is the justification
# for trusting them; see docs/frequency_domain_decision.md Section 1 for the full
# sourcing discussion and the residual risk this leaves.
SHAFT_RPM = 2000.0
SAMPLING_RATE_HZ = 20_000.0

N_ROLLERS = 16
PITCH_DIAMETER_IN = 2.815
ROLLER_DIAMETER_IN = 0.331
CONTACT_ANGLE_DEG = 15.17

# Half-width of the band summed around each characteristic frequency. The rig is
# belt-driven ("coupled to the shaft via rub belts", dataset PDF), so a small amount of
# belt slip / speed wander around the nominal 2000 RPM is expected; a hard single-bin
# read at exactly 236.40 Hz would fall off the peak. At 0.977 Hz resolution
# (20 kHz / 20,480), +/-3 Hz is ~7 bins -- wide enough to absorb that wander, narrow
# enough not to swallow neighbouring orders (BPFO and BPFI are ~60 Hz apart).
DEFECT_BAND_HALFWIDTH_HZ = 3.0

# Bearing defects excite a harmonic series, not just the fundamental, and the
# fundamental is often weaker than its harmonics under load. Summing the first three
# orders is the standard envelope/spectrum practice.
DEFECT_N_HARMONICS = 3

# STFT segment length for spectral kurtosis. 256 samples at 20 kHz gives 78 Hz bins and
# 160 frames per 20,480-point snapshot -- enough frames for a stable 4th-moment estimate
# per bin, which is what SK needs. SK localises *impulsiveness in a band*; it does not
# need fine frequency resolution (that is what the BPFO/BPFI band features are for).
SPECTRAL_KURTOSIS_NPERSEG = 256

# High-pass cutoff before envelope demodulation. Above the shaft orders and the
# low-frequency structural response, so what remains is the resonance-band ringing that
# bearing impacts actually modulate. 1 kHz is a conventional starting point rather than
# a derived value; docs/frequency_domain_decision.md records a sensitivity sweep.
ENVELOPE_HIGHPASS_HZ = 1000.0

FREQUENCY_DOMAIN_COLUMNS = [
    "experiment",
    "file_index",
    "timestamp",
    "bpfo_amplitude",
    "bpfi_amplitude",
    "bpfo_amplitude_norm",
    "bpfi_amplitude_norm",
    "bpfo_envelope_norm",
    "bpfi_envelope_norm",
    "spectral_kurtosis",
]


@dataclass(frozen=True)
class BearingFrequencies:
    """Characteristic defect frequencies (Hz) for one shaft speed."""

    shaft_hz: float
    bpfo_hz: float
    bpfi_hz: float


def compute_bearing_frequencies(
    shaft_rpm: float = SHAFT_RPM,
    n_rollers: int = N_ROLLERS,
    pitch_diameter: float = PITCH_DIAMETER_IN,
    roller_diameter: float = ROLLER_DIAMETER_IN,
    contact_angle_deg: float = CONTACT_ANGLE_DEG,
) -> BearingFrequencies:
    """Standard ball-pass defect frequencies from bearing geometry and shaft speed.

        BPFO = (n/2) * fr * (1 - (d/D) cos(theta))
        BPFI = (n/2) * fr * (1 + (d/D) cos(theta))

    where `fr` is shaft rotation frequency, `n` the roller count per row, `d` the roller
    diameter, `D` the pitch diameter, and `theta` the contact angle. `pitch_diameter` and
    `roller_diameter` need only be in consistent units -- only their ratio is used.

    With this module's defaults these give ~236.4 Hz / ~296.9 Hz, matching the
    independently published values for this dataset (see the sourcing note above).
    """
    shaft_hz = shaft_rpm / 60.0
    ratio = (roller_diameter / pitch_diameter) * math.cos(math.radians(contact_angle_deg))
    return BearingFrequencies(
        shaft_hz=shaft_hz,
        bpfo_hz=(n_rollers / 2.0) * shaft_hz * (1.0 - ratio),
        bpfi_hz=(n_rollers / 2.0) * shaft_hz * (1.0 + ratio),
    )


def compute_crest_factor(signal: np.ndarray) -> float:
    """Peak absolute amplitude divided by RMS. `nan` for an all-zero signal (RMS == 0),
    matching `03_feature_candidate_screening.ipynb`'s `peak / rms if rms > 0 else nan`."""
    rms = compute_rms(signal)
    if rms == 0:
        return float("nan")
    return float(np.max(np.abs(signal)) / rms)


def add_rolling_crest_factor(df: pd.DataFrame, rolling_window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add `crest_factor_smoothed`: a `rolling_window`-file rolling mean of
    `crest_factor`, `min_periods=1`.

    Not used by `extract_crest_factor`'s default output -- `docs/feature_windowing_decision.md`
    (Issue #40) treats crest factor as unwindowed by default, and Issue #23
    (`docs/skewness_crestfactor_decision.md`) found smoothing roughly doubles its
    separability but does not close the gap to kurtosis's, nor remove its redundancy
    with kurtosis in the experiment (`1st_test`) where it's otherwise most separable.
    Kept as a function, not deleted, alongside the rest of this evaluated-but-unused
    module.
    """
    out = df.copy()
    out["crest_factor_smoothed"] = out["crest_factor"].rolling(rolling_window, min_periods=1).mean()
    return out


def extract_crest_factor(
    raw_dir: Path,
    experiment: str,
    channel_idx: int,
) -> pd.DataFrame:
    """Compute per-file crest factor for one experiment.

    Same shape/style as `extraction.extract_experiment_features`, scoped to the one
    candidate feature evaluated and rejected in Issue #23.
    """
    files = list_snapshot_files(raw_dir)
    if not files:
        raise ValueError(f"No snapshot files found in {raw_dir}")

    records = []
    for i, f in enumerate(files):
        sig = load_channel(f, channel_idx)
        records.append(
            {
                "file_index": i,
                "timestamp": parse_timestamp(f),
                "crest_factor": compute_crest_factor(sig),
            }
        )

    df = pd.DataFrame.from_records(records)
    df["experiment"] = experiment
    return df[CREST_FACTOR_COLUMNS]


# --- Frequency-domain features (Issue #22) ----------------------------------------


def compute_amplitude_spectrum(
    signal: np.ndarray, sampling_rate: float = SAMPLING_RATE_HZ
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of a real signal, Hann-windowed.

    Returns `(freqs_hz, amplitude)`. Amplitude is normalised by signal length so it is
    comparable across snapshots of differing length (all 20,480 here, but the features
    below should not silently depend on that).

    A Hann window is applied first. Defect frequencies are not bin-centred (BPFO is
    236.40 Hz against a 0.977 Hz bin grid), so an unwindowed FFT leaks a rectangular
    window's slowly-decaying sidelobes across the spectrum: measured on a synthetic
    BPFO tone, ~1.7% of its amplitude lands in the BPFI band 60 Hz away. Since BPFO and
    BPFI are read as separate features, that leakage would make each partly a
    measurement of the other. Hann suppresses it by well over an order of magnitude
    (see `test_compute_band_amplitude_is_selective_to_the_requested_frequency`).

    The window's coherent gain (0.5 for Hann) scales all amplitudes by a constant, which
    cancels in `compute_band_amplitude_normalised` and is a harmless constant factor in
    the raw form -- no correction is applied, since no absolute-amplitude claim is made.
    """
    n = len(signal)
    windowed = signal * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return freqs, spectrum


def compute_band_amplitude(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    center_hz: float,
    halfwidth_hz: float = DEFECT_BAND_HALFWIDTH_HZ,
    n_harmonics: int = DEFECT_N_HARMONICS,
) -> float:
    """Total spectral amplitude in `+/- halfwidth_hz` bands around `center_hz` and its
    first `n_harmonics - 1` harmonics.

    Harmonics beyond Nyquist are skipped rather than wrapped. Returns the summed
    amplitude -- an absolute quantity that scales with overall vibration level, which is
    exactly why `compute_band_amplitude_normalised` exists alongside it (see Issue #22's
    redundancy finding).
    """
    total = 0.0
    for order in range(1, n_harmonics + 1):
        target = center_hz * order
        if target > freqs[-1]:
            break
        in_band = np.abs(freqs - target) <= halfwidth_hz
        total += float(spectrum[in_band].sum())
    return total


def compute_band_amplitude_normalised(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    center_hz: float,
    halfwidth_hz: float = DEFECT_BAND_HALFWIDTH_HZ,
    n_harmonics: int = DEFECT_N_HARMONICS,
) -> float:
    """`compute_band_amplitude` divided by the broadband noise floor (median amplitude
    across the whole spectrum).

    This is the physically meaningful "does a defect tone stand *out of* the noise?"
    measure, as opposed to "how loud is everything". The raw band amplitude rises
    whenever overall vibration rises, so on its own it largely restates RMS; dividing by
    the noise floor is what makes it a spectral-shape feature rather than an amplitude
    one. Both forms are computed so Issue #22 could measure that difference rather than
    assume it -- see `docs/frequency_domain_decision.md`.

    The median (not mean) is used as the noise-floor estimate so that the defect peaks
    themselves, and any strong shaft-order tones, do not inflate the denominator.
    """
    band = compute_band_amplitude(freqs, spectrum, center_hz, halfwidth_hz, n_harmonics)
    noise_floor = float(np.median(spectrum))
    if noise_floor == 0:
        return float("nan")
    return band / noise_floor


def compute_envelope_spectrum(
    signal: np.ndarray,
    sampling_rate: float = SAMPLING_RATE_HZ,
    highpass_hz: float = ENVELOPE_HIGHPASS_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope (demodulation) spectrum: the standard way to surface bearing defect
    frequencies, and the reason a plain-spectrum null result would not be conclusive.

    A rolling-element defect does not usually inject energy *at* 236 Hz. Each impact
    rings the structure's high-frequency resonances, producing a burst train that
    repeats at the defect rate -- so the defect frequency appears as the *modulation
    rate* of a high-frequency carrier, not as a line in the raw spectrum. Reading the
    raw spectrum at BPFO can therefore find nothing even when the defect is plainly
    present, which is exactly the trap this function exists to avoid: any "frequency-
    domain features don't work here" conclusion has to survive envelope analysis too,
    not just the plain FFT.

    Steps: high-pass above `highpass_hz` to discard shaft orders and low-frequency
    structural response, Hilbert-transform to get the analytic signal, take its
    magnitude (the envelope), remove the envelope's DC offset (otherwise bin 0 dwarfs
    everything), and FFT the result.

    The high-pass cutoff is a judgement call, not a derived constant -- see
    `docs/frequency_domain_decision.md` for the sensitivity check across cutoffs.
    """
    from scipy.signal import butter, filtfilt, hilbert

    nyquist = sampling_rate / 2.0
    if not 0 < highpass_hz < nyquist:
        raise ValueError(f"highpass_hz must be in (0, {nyquist}), got {highpass_hz}")

    b, a = butter(4, highpass_hz / nyquist, btype="highpass")
    filtered = filtfilt(b, a, signal)

    envelope = np.abs(hilbert(filtered))
    envelope = envelope - envelope.mean()

    n = len(envelope)
    spectrum = np.abs(np.fft.rfft(envelope * np.hanning(n))) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return freqs, spectrum


def compute_spectral_kurtosis(
    signal: np.ndarray,
    sampling_rate: float = SAMPLING_RATE_HZ,
    nperseg: int = SPECTRAL_KURTOSIS_NPERSEG,
) -> float:
    """Maximum spectral kurtosis across frequency bins (Antoni's SK indicator).

    Per frequency bin, over STFT time frames:

        SK(f) = E[|X(t,f)|^4] / E[|X(t,f)|^2]^2 - 2

    The `-2` makes SK zero for a stationary complex-Gaussian process, so positive SK
    marks bins where energy arrives in bursts rather than steadily. Summarised as the
    max over bins -- the standard single-number SK indicator, and the form that answers
    "is there *some* band behaving impulsively", which is the question a bearing defect
    poses.

    Complementary to time-domain kurtosis by construction: time-domain kurtosis asks
    whether the *waveform* has heavy tails overall, SK asks whether any *individual
    frequency band* is bursty even when the broadband signal is not. Whether that
    distinction actually buys anything on this dataset is Issue #22's question, answered
    in `docs/frequency_domain_decision.md`.
    """
    from scipy.signal import stft

    _, _, zxx = stft(signal, fs=sampling_rate, nperseg=nperseg)
    power = np.abs(zxx) ** 2  # |X(t,f)|^2, shape (n_freqs, n_frames)

    mean_sq = power.mean(axis=1)
    mean_quad = (power**2).mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sk = np.where(mean_sq > 0, mean_quad / (mean_sq**2) - 2.0, np.nan)

    if np.all(np.isnan(sk)):
        return float("nan")
    return float(np.nanmax(sk))


def extract_frequency_domain_features(
    raw_dir: Path,
    experiment: str,
    channel_idx: int,
    sampling_rate: float = SAMPLING_RATE_HZ,
    frequencies: BearingFrequencies | None = None,
) -> pd.DataFrame:
    """Compute per-file BPFO/BPFI band amplitudes and spectral kurtosis for one experiment.

    Both BPFO and BPFI are computed for **every** experiment, regardless of that
    experiment's documented failure mode. Two reasons: it keeps a single code path with
    no test-set-specific branching (the same constraint Issue #43 imposed on
    `extract_experiment_features`), and it lets the evaluation *check* whether the
    defect frequency matching the documented fault is the one that actually responds,
    rather than assuming it.
    """
    files = list_snapshot_files(raw_dir)
    if not files:
        raise ValueError(f"No snapshot files found in {raw_dir}")

    freqs_ref = frequencies or compute_bearing_frequencies()

    records = []
    for i, f in enumerate(files):
        sig = load_channel(f, channel_idx)
        freqs, spectrum = compute_amplitude_spectrum(sig, sampling_rate)
        env_freqs, env_spectrum = compute_envelope_spectrum(sig, sampling_rate)
        records.append(
            {
                "file_index": i,
                "timestamp": parse_timestamp(f),
                "bpfo_amplitude": compute_band_amplitude(freqs, spectrum, freqs_ref.bpfo_hz),
                "bpfi_amplitude": compute_band_amplitude(freqs, spectrum, freqs_ref.bpfi_hz),
                "bpfo_amplitude_norm": compute_band_amplitude_normalised(
                    freqs, spectrum, freqs_ref.bpfo_hz
                ),
                "bpfi_amplitude_norm": compute_band_amplitude_normalised(
                    freqs, spectrum, freqs_ref.bpfi_hz
                ),
                "bpfo_envelope_norm": compute_band_amplitude_normalised(
                    env_freqs, env_spectrum, freqs_ref.bpfo_hz
                ),
                "bpfi_envelope_norm": compute_band_amplitude_normalised(
                    env_freqs, env_spectrum, freqs_ref.bpfi_hz
                ),
                "spectral_kurtosis": compute_spectral_kurtosis(sig, sampling_rate),
            }
        )

    df = pd.DataFrame.from_records(records)
    df["experiment"] = experiment
    return df[FREQUENCY_DOMAIN_COLUMNS]
