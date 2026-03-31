import os
import numpy as np
import pygame
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

_client      = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_WARM   = os.getenv("ELEVENLABS_VOICE_WARM",   "EXAVITQu4vr4xnSDxMaL")
VOICE_DIRECT = os.getenv("ELEVENLABS_VOICE_DIRECT", "IKne3meq5aSn9XLyUdCD")

pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=512)

def _get_voice_settings(adapted_ipc: dict, archetype: str) -> dict:
    """
    Gemini was correct — eleven_flash_v2_5 does NOT support SSML tags.
    Emotion is controlled entirely through voice_settings parameters.

    stability:        lower = more expressive/emotional, higher = consistent
    similarity_boost: how closely to match the original voice
    style:            expressiveness level (0 = neutral, 1 = very expressive)
    speed:            speaking rate
    """
    dom  = adapted_ipc.get("dominance", 0.5)
    warm = adapted_ipc.get("warmth",    0.6)
    pace = adapted_ipc.get("pace",      "medium")
    giving_up = adapted_ipc.get("giving_up", False)

    # Speed from pace
    speed_map = {"slow": 0.85, "medium": 1.0, "fast": 1.10}
    speed     = speed_map.get(pace, 1.0)

    if giving_up:
        # Maximum warmth, very gentle, slow
        return {
            "stability":         0.35,   # more emotional variance
            "similarity_boost":  0.80,
            "style":             0.70,   # very expressive
            "speed":             0.67,
        }

    if archetype == "maya":
        # Warm, encouraging, patient
        return {
            "stability":         0.40,   # some warmth variation
            "similarity_boost":  0.80,
            "style":             0.55,   # moderately expressive
            "speed":             speed * 0.93,  # slightly slower
        }
    elif archetype == "arjun":
        # Direct, confident, crisp
        return {
            "stability":         0.65,   # consistent, controlled
            "similarity_boost":  0.85,
            "style":             0.25,   # less expressive, more authoritative
            "speed":             speed * 1.05,
        }
    else:
        # Lina — calm, clear, balanced
        return {
            "stability":         0.50,
            "similarity_boost":  0.82,
            "style":             0.40,
            "speed":             speed,
        }

def _get_voice_id(archetype: str) -> str:
    if archetype in ("maya", "lina"):
        return VOICE_WARM
    return VOICE_DIRECT

def stream_tts_and_play(text: str,
                         adapted_ipc: dict,
                         archetype: str):
    """
    Stream ElevenLabs Flash v2.5 and play PCM chunks as they arrive.
    No SSML tags — emotion via voice_settings only (Gemini was correct).
    """
    if not text.strip():
        return

    voice_id       = _get_voice_id(archetype)
    voice_settings = _get_voice_settings(adapted_ipc, archetype)

    try:
        audio_stream = _client.text_to_speech.stream(
            text=text,
            voice_id=voice_id,
            model_id="eleven_flash_v2_5",
            output_format="pcm_16000",
            voice_settings=voice_settings,
        )

        for chunk in audio_stream:
            if chunk:
                # 1. Convert the raw buffer to a numpy array
                arr = np.frombuffer(chunk, dtype=np.int16)
                
                # 2. Check if the mixer is in stereo mode (channels == 2)
                # pygame.mixer.get_init() returns (frequency, format, channels)
                if pygame.mixer.get_init()[2] == 2:
                    # Duplicate the mono channel into two columns for stereo
                    arr = np.column_stack((arr, arr))
                    
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
        
                while pygame.mixer.get_busy():
                     pygame.time.Clock().tick(40)

    except Exception as e:
        print(f"  [TTS error: {e}]")