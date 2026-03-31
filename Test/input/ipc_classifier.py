import librosa
import numpy as np
from typing import Optional

def extract_ipc_vector(audio_path: str,
                        prev_dominance: Optional[float] = None,
                        filler_count: int = 0,
                        response_latency_ms: float = 0.0) -> dict:
    """
    Extract 8-signal IPC vector from audio file.
    Returns dict with all IPC features.
    """
    y, sr = librosa.load(audio_path, sr=16000)

    # ── Pitch ───────────────────────────────────────────────────────
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=400)
    f0_clean  = f0[~np.isnan(f0)]
    pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 150.0
    pitch_var  = float(np.std(f0_clean))  if len(f0_clean) > 0 else 20.0

    # ── Energy / volume ─────────────────────────────────────────────
    rms            = librosa.feature.rms(y=y)[0]
    mean_rms       = float(np.mean(rms))
    speech_frames  = np.sum(rms > 0.02)
    total_frames   = len(rms)
    speech_ratio   = speech_frames / total_frames if total_frames > 0 else 0.5
    pause_ratio    = 1.0 - speech_ratio

    # Volume drop mid-sentence: compare first vs second half RMS
    mid             = len(rms) // 2
    first_half_rms  = float(np.mean(rms[:mid])) if mid > 0 else mean_rms
    second_half_rms = float(np.mean(rms[mid:])) if mid > 0 else mean_rms
    volume_drop     = (first_half_rms - second_half_rms) > 0.015

    # ── IPC scores ──────────────────────────────────────────────────
    dominance = float(np.clip(
        (speech_ratio - 0.4) * 2.0 + (mean_rms - 0.05) * 10, 0, 1
    ))
    warmth    = float(np.clip((pitch_var - 10) / 60, 0, 1))

    # Pace
    if speech_ratio > 0.70:
        pace = "fast"
    elif speech_ratio > 0.50:
        pace = "medium"
    else:
        pace = "slow"

    # ── Derived signals ─────────────────────────────────────────────
    # Assertiveness delta vs previous turn
    assertiveness_delta = 0.0
    if prev_dominance is not None:
        assertiveness_delta = round(dominance - prev_dominance, 3)

    # Giving-up: very short utterance + high pause
    giving_up = (speech_ratio < 0.25 and pause_ratio > 0.70)

    # Enthusiasm: rising pitch in second half + long utterance
    enthusiasm = (pitch_var > 35 and speech_ratio > 0.65)

    # Social comfort proxy (inverse of filler density)
    words_approx     = max(1, int(speech_ratio * 100))
    filler_density   = filler_count / words_approx
    social_comfort   = float(np.clip(1.0 - filler_density * 5, 0, 1))

    # Question vs statement (rough proxy from pitch ending)
    # Rising pitch at end = question
    if len(f0_clean) > 10:
        pitch_end   = float(np.mean(f0_clean[-5:]))
        pitch_start = float(np.mean(f0_clean[:5]))
        is_question = pitch_end > pitch_start * 1.15
    else:
        is_question = False

    return {
        # Core IPC
        "dominance":           round(dominance, 3),
        "warmth":              round(warmth, 3),
        "pace":                pace,
        # Extended signals
        "pause_ratio":         round(pause_ratio, 3),
        "pitch_variance":      round(pitch_var, 2),
        "assertiveness_delta": assertiveness_delta,
        "giving_up":           giving_up,
        "enthusiasm":          enthusiasm,
        "social_comfort":      round(social_comfort, 3),
        "volume_drop":         volume_drop,
        "is_question":         is_question,
        "filler_count":        filler_count,
        "response_latency_ms": response_latency_ms,
    }