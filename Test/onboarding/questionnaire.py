import asyncio
import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from input.transcriber import transcribe
from input.ipc_classifier import extract_ipc_vector
from input.recorder import record_until_enter, save_wav
from output.tts_client import stream_tts_and_play

# 🔥 FIXED: Import the dedicated signal extractor logic
from onboarding.signal_extractor import extract_signals_single

load_dotenv()
logger = logging.getLogger("SYRA.Onboarding")

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

QUESTIONS = [
    ("q1",  "Which subject do you find the most difficult, and what makes it hard for you?"),
    ("q2",  "When you get something wrong, what do you usually do — try again, ask someone, or move on?"),
    ("q3",  "When learning something new, do you prefer to understand the big picture first, or go step by step?"),
    ("q4",  "Does it help when I use stories or real-life examples, or do you prefer direct explanations?"),
    ("q5",  "When you're still figuring something out, does being asked questions feel helpful or like pressure?"),
    ("q6",  "What's your main goal right now — board exams, JEE or NEET, or just understanding better?"),
    ("q7",  "If a teacher explains something and you don't get it, do you ask immediately, wait, or figure it out yourself?"),
    ("q8",  "Tell me something you feel really confident about — explain it like I have no idea what it is."),
    ("q9",  "How much encouragement do you like — a lot, some, or just keep things focused?"),
    ("q10", "Do you prefer covering many topics quickly, or going really deep on one thing?"),
    ("q11", "Is there anything in any subject where your understanding might be a little off?"),
    ("q12", "If you could tell every teacher one thing about how you learn best — what would it be?"),
]

IPC_RICH = {"q1", "q8", "q12"}

# Navigation Patterns
REPEAT_PATTERNS = ["repeat", "say that again", "again", "what was the question", "can you repeat", "pardon"]
PREVIOUS_PATTERNS = ["previous question", "last question", "go back", "before that"]
SKIP_PATTERNS = ["skip", "next question", "move on", "pass"]

def _detect_command(text: str) -> Optional[str]:
    text_lower = text.lower().strip()
    if any(p in text_lower for p in REPEAT_PATTERNS): return "repeat"
    if any(p in text_lower for p in PREVIOUS_PATTERNS): return "previous"
    if any(p in text_lower for p in SKIP_PATTERNS): return "skip"
    return None

# ── Helpers ───────────────────────────────────────────────────────────
def _load_json_safely(path: Path) -> Dict[str, Any]:
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}

def _most_common(values: List[Any]) -> Any:
    cleaned = [v for v in values if v and v != "null"]
    return max(set(cleaned), key=cleaned.count) if cleaned else None

def _save_checkpoint(path: Path, signals: list, ipc: list):
    draft = {
        "onboarding_done": False,
        "_partial_signals": signals,
        "_partial_ipc": ipc,
        "_checkpoint_time": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(draft, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

# ── Profile builder ───────────────────────────────────────────────────
def _build_profile(all_signals: List[dict], all_ipc: List[dict], student_id: str) -> dict:
    dom_vals  = [v.get("dominance", 0.5) for v in all_ipc if v]
    warm_vals = [v.get("warmth", 0.6) for v in all_ipc if v]
    paces     = [v.get("pace") for v in all_ipc if v and v.get("pace")]

    avg_dom  = round(sum(dom_vals) / len(dom_vals), 3) if dom_vals else 0.5
    avg_warm = round(sum(warm_vals) / len(warm_vals), 3) if warm_vals else 0.6

    if avg_dom < 0.40 and avg_warm > 0.60: archetype = "maya"
    elif avg_dom > 0.60: archetype = "arjun"
    else: archetype = "lina"

    processing = _most_common([s.get("processing_style") for s in all_signals])
    chunk_size = 1 if processing == "bottom_up" else 3 if processing == "top_down" else 2

    return {
        "student_id": student_id,
        "onboarding_done": True,
        "onboarding_date": datetime.now().isoformat(),
        "ipc": {
            "dominance": avg_dom,
            "warmth": avg_warm,
            "archetype": archetype,
            "pace": _most_common(paces) or "medium",
            "assertiveness_delta": 0.0,
        },
        "learning_style": {
            "processing_style": processing,
            "analogy_receptiveness": _most_common([s.get("analogy_receptiveness") for s in all_signals]),
            "goal_type": _most_common([s.get("goal_type") for s in all_signals]) or "boards",
            "encouragement_need": _most_common([s.get("encouragement_need") for s in all_signals]),
        },
        "cognitive_load": {"chunk_size": chunk_size},
        "knowledge": {"mastery_map": {}, "misconceptions": []},
        "session_history": []
    }

# ── Single question handler ───────────────────────────────────────────
def _ask_question(q_id: str, q_text: str, neutral_ipc: dict, prev_q_text: Optional[str] = None) -> tuple[dict, dict]:
    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        attempt += 1
        input("   [ Press Enter to answer ]")
        audio_data = record_until_enter()
        wav_path = save_wav(audio_data)
        answer, fillers, _ = transcribe(wav_path)

        if not answer.strip():
            stream_tts_and_play("I didn't catch that. Let me ask again.", neutral_ipc, "lina")
            stream_tts_and_play(q_text, neutral_ipc, "lina")
            continue

        print(f"   Student: {answer}")
        command = _detect_command(answer)

        if command == "repeat":
            stream_tts_and_play("Of course! " + q_text, neutral_ipc, "lina")
            continue
        if command == "previous":
            raise _NavigationCommand("previous")
        if command == "skip":
            raise _NavigationCommand("skip")

        # Process real answer
        ipc = None
        if q_id in IPC_RICH:
            ipc = extract_ipc_vector(wav_path, filler_count=fillers)
        
        if os.path.exists(wav_path): os.unlink(wav_path)

        # 🔥 FIXED: Using the external signal extractor which handles the Gemini API correctly
        signals = extract_signals_single(q_text, answer)

        stream_tts_and_play(random.choice(["Got it.", "Makes sense.", "Interesting."]), neutral_ipc, "lina")
        return signals, (ipc if ipc else neutral_ipc)

    return {}, neutral_ipc

class _NavigationCommand(Exception):
    pass

# ── Main entry ────────────────────────────────────────────────────────
def run_onboarding(student_id: str, pm=None) -> dict:
    profile_path = Path(f"sessions/{student_id}/student_profile.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_json_safely(profile_path)
    if existing.get("onboarding_done"):
        return existing

    all_signals = existing.get("_partial_signals", [])
    all_ipc = existing.get("_partial_ipc", [])
    i = len(all_signals)
    neutral_ipc = {"dominance": 0.5, "warmth": 0.7, "pace": "medium", "giving_up": False}

    if i == 0:
        stream_tts_and_play("Hi! I'm SYRA. Let's do twelve quick questions to see how you learn best.", neutral_ipc, "lina")

    while i < len(QUESTIONS):
        q_id, q_text = QUESTIONS[i]
        prev_q_text = QUESTIONS[i - 1][1] if i > 0 else None
        stream_tts_and_play(q_text, neutral_ipc, "lina")

        try:
            signals, ipc = _ask_question(q_id, q_text, neutral_ipc, prev_q_text)
            if i < len(all_signals):
                all_signals[i], all_ipc[i] = signals, ipc
            else:
                all_signals.append(signals); all_ipc.append(ipc)
            
            _save_checkpoint(profile_path, all_signals, all_ipc)
            i += 1
        except _NavigationCommand as cmd:
            if cmd.args[0] == "previous" and i > 0:
                i -= 1; all_signals = all_signals[:i]; all_ipc = all_ipc[:i]
            elif cmd.args[0] == "skip":
                all_signals.append({}); all_ipc.append(neutral_ipc)
                i += 1
            _save_checkpoint(profile_path, all_signals, all_ipc)

    profile = _build_profile(all_signals, all_ipc, student_id)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4, ensure_ascii=False)
    
    stream_tts_and_play("All set! Let's get started.", neutral_ipc, profile["ipc"]["archetype"])
    return profile