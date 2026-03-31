import os
import httpx
import json
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL     = "https://api.deepgram.com/v1/listen"

def transcribe(audio_path: str) -> tuple[str, int, float]:
    """
    Transcribe audio using Deepgram Nova-3.
    Uses raw HTTP instead of SDK — works with any SDK version
    including v6.x which restructured the client completely.
    Returns: (transcript, filler_count, duration_seconds)
    """
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    params = {
        "model":        "nova-3",
        "language":     "en-IN",
        "smart_format": "true",
        "punctuate":    "true",
        "filler_words": "true",
        "utterances":   "true",
    }

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type":  "audio/wav",
    }

    try:
        response = httpx.post(
            DEEPGRAM_URL,
            params=params,
            headers=headers,
            content=audio_data,
            timeout=30.0,
        )
        response.raise_for_status()
        result = response.json()

    except httpx.HTTPStatusError as e:
        print(f"  [Deepgram HTTP error: {e.response.status_code}]")
        return "", 0, 0.0
    except Exception as e:
        print(f"  [Deepgram error: {e}]")
        return "", 0, 0.0

    # ── Parse ─────────────────────────────────────────────────────────
    try:
        channel    = result["results"]["channels"][0]
        alt        = channel["alternatives"][0]
        transcript = alt.get("transcript", "").strip()
        words      = alt.get("words", [])
        fillers    = sum(1 for w in words
                        if w.get("type") == "filler")
        duration   = result.get("metadata", {}).get("duration", 0.0)

        return transcript, fillers, duration

    except (KeyError, IndexError) as e:
        print(f"  [Deepgram parse error: {e}]")
        print(f"  Raw response: {json.dumps(result, indent=2)[:300]}")
        return "", 0, 0.0

    