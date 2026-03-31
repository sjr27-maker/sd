import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os

SAMPLE_RATE  = 16000

def get_device_index() -> int:
    idx = os.getenv("DEVICE_INDEX", "1")
    return int(idx)

def record_until_enter() -> np.ndarray:
    """Record audio until user presses Enter. Returns float32 numpy array."""
    print("\n  🎤 Speak now... (Press Enter to stop)")
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=get_device_index(),
        callback=callback
    ):
        input()

    if not frames:
        return np.zeros(SAMPLE_RATE, dtype="float32")
    return np.concatenate(frames).flatten()

def save_wav(audio_np: np.ndarray) -> str:
    """Save float32 audio to a temp WAV file. Returns file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE,
              (audio_np * 32767).astype(np.int16))
    return tmp.name