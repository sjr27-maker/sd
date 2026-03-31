import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

from input.transcriber import transcribe
from input.ipc_classifier import extract_ipc_vector
from input.recorder import record_until_enter, save_wav
from output.tts_client import stream_tts_and_play

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SYRA.Onboarding")

# 2026 Standard: Using Gemini 3 Flash
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash" 

QUESTIONS = [
    ("q1",  "Which subject do you find the most difficult, and what makes it hard for you?"),
    ("q2",  "When you get something wrong, what do you usually do — try again, ask someone, or move on?"),
    ("q3",  "When learning something new, do you prefer to understand the big picture first, or go step by step?"),
    ("q4",  "Does it help when I use stories or real-life examples, or do you prefer direct explanations?"),
    ("q5",  "When you're still figuring something out, does being asked questions feel helpful or like pressure?"),
    ("q6",  "What's your main goal right now — board exams, JEE or NEET, or just understanding better?"),
    ("q7",  "If a teacher explains something and you don't get it, do you ask immediately, wait, or figure it out yourself?"),
    ("q8",  "Tell me something you feel really confident about — explain it to me like I have no idea what it is."),
    ("q9",  "How much encouragement do you like — a lot, some, or just keep things focused on the work?"),
    ("q10", "Do you prefer covering many topics quickly, or going really deep on one thing before moving on?"),
    ("q11", "Is there anything in any subject where you think your understanding might be a little off or incomplete?"),
    ("q12", "If you could tell every teacher one thing about how you learn best — what would it be?"),
]

IPC_RICH = {"q1", "q8", "q12"}

# --- Helpers ---

def _load_json_safely(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except:
        return {}

def _most_common(values: List[Any]) -> Any:
    cleaned = [v for v in values if v and v != "null"]
    return max(set(cleaned), key=cleaned.count) if cleaned else None

# --- Signal Extraction ---

def _extract_signals(question: str, answer: str) -> Dict[str, Any]:
    """Uses Gemini 3 Flash to extract learning signals."""
    prompt = f"Question: {question}\nAnswer: {answer}\n\nExtract signals as JSON..." # (Use your full prompt here)
    try:
        response = _client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Signal extraction failed: {e}")
        return {}

# --- Final Profile Assembler ---

def _build_profile(all_signals: List[dict], all_ipc: List[dict], student_id: str) -> dict:
    """Consolidates the 12 answers into the final student_profile.json."""
    
    # Calculate Averages for the final profile
    dom_vals  = [v.get("dominance", 0.5) for v in all_ipc if v]
    warm_vals = [v.get("warmth", 0.6) for v in all_ipc if v]
    
    avg_dom  = round(sum(dom_vals) / len(dom_vals), 3) if dom_vals else 0.5
    avg_warm = round(sum(warm_vals) / len(warm_vals), 3) if warm_vals else 0.6

    # Archetype Logic
    if avg_dom < 0.45 and avg_warm > 0.65: archetype = "maya"
    elif avg_dom > 0.65: archetype = "arjun"
    else: archetype = "lina"

    profile = {
        "student_id": student_id,
        "onboarding_done": True,
        "onboarding_date": datetime.now().isoformat(),
        "ipc": {
            "dominance": avg_dom,
            "warmth": avg_warm,
            "archetype": archetype,
            "pace": _most_common([v.get("pace") for v in all_ipc if v]) or "medium"
        },
        "learning_style": {
            "processing_style": _most_common([s.get("processing_style") for s in all_signals]),
            "goal_type": _most_common([s.get("goal_type") for s in all_signals]) or "boards"
        },
        "knowledge": {
            "mastery_map": {},
            "misconceptions": [s["misconception_seed"] for s in all_signals if s.get("misconception_seed")]
        },
        "session_history": []
    }
    return profile

# --- Main Logic ---

def run_onboarding(student_id: str, pm=None) -> dict:
    profile_path = Path(f"sessions/{student_id}/student_profile.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Check for existing or partial profile
    existing = _load_json_safely(profile_path)
    
    if existing.get("onboarding_done"):
        print(f"  [SYRA] Profile loaded for {student_id}")
        return existing

    # ── RESUME LOGIC ──
    # If the file exists but isn't 'done', we resume from the last recorded answer
    all_signals = existing.get("_partial_signals", [])
    all_ipc = existing.get("_partial_ipc", [])
    start_index = len(all_signals)

    if start_index > 0:
        print(f"  [SYRA] Resuming onboarding from question {start_index + 1}...")
    else:
        print("\n" + "="*54 + "\n  SYRA — First-time setup\n" + "="*54 + "\n")
        stream_tts_and_play("Hi! I'm SYRA. Let's do a quick check-in.", {"dominance":0.5, "warmth":0.7}, "lina")

    # 2. Loop through remaining questions
    for i in range(start_index, len(QUESTIONS)):
        q_id, q_text = QUESTIONS[i]
        print(f"\n[{q_id.upper()}] SYRA: {q_text}")
        stream_tts_and_play(q_text, {"dominance":0.5, "warmth":0.7}, "lina")

        input("  [ Press Enter to answer ]")
        audio_data = record_until_enter()
        wav_path = save_wav(audio_data)

        # Transcribe
        answer, fillers, _ = transcribe(wav_path)
        if not answer.strip():
            answer = "No response"
            
        print(f"  Student: {answer}")

        # IPC & Signal Analysis
        ipc = None
        if q_id in IPC_RICH:
            ipc = extract_ipc_vector(wav_path, filler_count=fillers)
        
        signals = _extract_signals(q_text, answer)
        
        if os.path.exists(wav_path):
            os.unlink(wav_path)

        # ── THE CHECKPOINT SAVE ──
        all_signals.append(signals)
        all_ipc.append(ipc if ipc else {"dominance": 0.5, "warmth": 0.6, "pace": "medium"})
        
        # We save a temporary "draft" version of the profile
        draft_data = {
            "onboarding_done": False,
            "_partial_signals": all_signals,
            "_partial_ipc": all_ipc
        }
        profile_path.write_text(json.dumps(draft_data, indent=2))
        
        ack = random.choice(["Got it.", "Okay.", "Makes sense."])
        stream_tts_and_play(ack, {"dominance":0.5, "warmth":0.7}, "lina")

    # 3. Finalization (Only runs if ALL questions finished)
    print("\n  All questions answered. Assembling final profile...")
    final_profile = _build_profile(all_signals, all_ipc, student_id)
    
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(final_profile, f, indent=4, ensure_ascii=False)

    print(f"✅ Onboarding Complete. Profile saved.")
    
    stream_tts_and_play("All set! Let's get started.", final_profile["ipc"], final_profile["ipc"]["archetype"])
    
    return final_profile