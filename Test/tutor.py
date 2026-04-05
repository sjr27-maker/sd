import os
import json
import time
import logging
import threading
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import scipy.io.wavfile as wav
from dotenv import load_dotenv

from input.recorder          import record_until_enter, save_wav
from input.transcriber       import transcribe
from input.ipc_classifier    import extract_ipc_vector

from brain.layer1_personality   import get_ipc_style_instructions, infer_archetype
from brain.layer2_knowledge     import infer_mastery_from_exchange
from brain.layer4_emotional     import detect_emotional_state, get_emotional_instruction
from brain.layer6_cognitive_load import get_load_instruction, update_confusion_counter
from brain.layer7_prerequisite  import get_prerequisite_instruction
from brain.layer8_comprehension import (
    should_trigger_teach_back,
    get_teach_back_instruction,
    score_session_comprehension,
)
from brain.layer_style_mirror import (
    extract_session_style,
    update_style_profile,
    get_mirror_instruction,
)

from engine.prompt_builder   import build_system_prompt
from engine.llm_client       import stream_response
from engine.rag_retriever    import retrieve_context

from memory.profile_manager  import ProfileManager
from memory.session_memory   import SessionMemory

from output.tts_client       import stream_tts_and_play
from onboarding.questionnaire import run_onboarding

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SYRA.Tutor")

SAMPLE_RATE = 16000


# ── Session end ───────────────────────────────────────────────────────

def _end_session(pm:         ProfileManager,
                  sm:         SessionMemory,
                  profile:    dict,
                  subject:    str,
                  grade:      int,
                  student_id: str,
                  style_profile: dict):
    """
    Score comprehension, update all memory layers, save session log.
    Called on Ctrl+C or normal session end.
    """
    if not sm.turns:
        print("  No turns recorded — skipping save.")
        return

    print("\n  Scoring session...")

    # Comprehension scoring — one Gemini call on full transcript
    session_data = score_session_comprehension(sm.turns, subject)
    print(f"  Comprehension : {session_data.get('comprehension_score', 0)}/100")
    print(f"  Topics covered: {session_data.get('topics_covered', [])}")
    print(f"  Struggling    : {session_data.get('topics_struggling', [])}")

    # IPC summary across all turns
    dom_vals  = [t["ipc_vector"]["dominance"]
                 for t in sm.turns if t.get("ipc_vector")]
    warm_vals = [t["ipc_vector"]["warmth"]
                 for t in sm.turns if t.get("ipc_vector")]
    fill_vals = [t["ipc_vector"].get("filler_count", 0)
                 for t in sm.turns if t.get("ipc_vector")]

    ipc_summary = {
        "avg_dominance":    round(sum(dom_vals)  / len(dom_vals),  3)
                            if dom_vals  else 0.5,
        "avg_warmth":       round(sum(warm_vals) / len(warm_vals), 3)
                            if warm_vals else 0.6,
        "avg_filler_count": round(sum(fill_vals) / len(fill_vals), 1)
                            if fill_vals else 0.0,
    }

    # Style profile — batch extraction, more accurate than per-turn
    print("  Updating style profile...")
    new_style     = extract_session_style(sm.turns)
    updated_style = update_style_profile(style_profile, new_style)

    # Gated base profile update (drift protection)
    pm.update_base_profile(ipc_summary, len(sm.turns))

    # Fast knowledge updates — no gates needed
    pm.update_knowledge(
        mastery_updates={},     # updated per-turn in background
        new_misconceptions=session_data.get("new_misconceptions", []),
        comprehension_score=session_data.get("comprehension_score", 0),
        topics_covered=session_data.get("topics_covered", []),
        topics_struggling=session_data.get("topics_struggling", []),
        recommended_next=session_data.get("recommended_next_topic"),
    )

    # Save style profile to permanent memory
    pm.profile["style_profile"] = updated_style
    pm.save()

    slang_count = len(updated_style.get("confirmed_slang", []))
    print(f"  Style profile : {slang_count} confirmed slang terms")
    print(f"  α next session: {pm.get_alpha():.2f}")
    print(f"  Archetype     : {pm.profile['ipc']['archetype']}")

    # Save full session log
    log = sm.to_log()
    log.update(session_data)
    log["ipc_summary"] = ipc_summary
    log["grade"]       = grade
    log["subject"]     = subject

    out_dir  = Path(f"sessions/{student_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / \
        f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"  Session saved → {out_path}")


# ── Background mastery update ─────────────────────────────────────────

def _update_mastery_bg(student_text: str,
                        ai_text:      str,
                        topic:        str,
                        profile:      dict):
    """
    Runs in background thread after each turn.
    Infers mastery level from the exchange and updates profile.
    Does not block the conversation.
    """
    try:
        result  = infer_mastery_from_exchange(student_text, ai_text, topic)
        mastery = result.get("mastery_level", "introduced")
        now     = datetime.now().isoformat()

        profile.setdefault(
            "knowledge", {}
        ).setdefault("mastery_map", {})[topic] = {
            "mastery":   mastery,
            "last_seen": now,
        }

        if result.get("misconception_found") and result.get("misconception_desc"):
            existing = profile.get("mental_model", {}).get("naive_theories", [])
            descs    = [m.get("description", "") if isinstance(m, dict) else m
                        for m in existing]
            desc = result["misconception_desc"]
            if desc not in descs:
                existing.append({
                    "description": desc,
                    "resolved":    False,
                    "found_date":  now,
                })
                profile.setdefault("mental_model", {})["naive_theories"] = existing

    except Exception as e:
        logger.debug(f"Mastery update error: {e}")


# ── Main session loop ─────────────────────────────────────────────────

def run_session(subject:    str = "Mathematics",
                grade:      int = 9,
                student_id: str = "student_001"):
    """
    Half-duplex tutoring session — stable, fully tested path.
    Student presses Enter to speak, Enter again to stop.
    Every turn: STT → IPC → prompt build → LLM stream → TTS → log.
    """

    # ── Setup ─────────────────────────────────────────────────────────
    pm      = ProfileManager(student_id)
    profile = run_onboarding(student_id, pm)
    sm      = SessionMemory(profile, subject)

    archetype     = profile["ipc"]["archetype"]
    style_profile = profile.get("style_profile", {})
    session_count = len(profile.get("session_history", []))

    # Session state — mirrors live_session exactly
    conversation_history : list         = []
    consecutive_confused : int          = 0
    current_topic        : Optional[str]= None
    prev_dominance       : float        = profile["ipc"]["dominance"]
    turn_num             : int          = 0

    print(f"\n{'='*54}")
    print(f"  SYRA Tutor | Class {grade} {subject} | {archetype.upper()}")
    print(f"  Press Enter → speak → Press Enter to stop")
    print(f"  Ctrl+C to end session")
    print(f"{'='*54}\n")

    # ── Opening ───────────────────────────────────────────────────────
    ctx = sm.context

    if ctx.get("returning") and ctx.get("last_session"):
        last = ctx["last_session"]
        rec  = last.get("recommended_next_topic", "")
        comp = last.get("comprehension_score", "?")
        opening = (
            f"Welcome back! Last time your comprehension score was {comp}. "
            + (f"Shall we continue with {rec}?"
               if rec else "What would you like to work on today?")
        )
    else:
        openings = {
            "maya":  f"Let's go through {subject} together. "
                     f"What would you like to start with?",
            "arjun": f"Let's get into {subject}. What are we working on?",
            "lina":  f"Let's get started with {subject}. "
                     f"Which topic today?",
        }
        opening = openings.get(archetype, openings["lina"])

    # Seed IPC for opening TTS
    seed_ipc = {
        "dominance":  profile["ipc"]["dominance"],
        "warmth":     profile["ipc"]["warmth"],
        "pace":       profile["ipc"]["pace"],
        "giving_up":  False,
        "filler_count": 0,
    }

    print(f"  SYRA: {opening}")
    stream_tts_and_play(opening, seed_ipc, archetype)

    # ── Main loop ─────────────────────────────────────────────────────
    try:
        while True:
            turn_num += 1
            print(f"\n── Turn {turn_num} {'─'*38}")
            input("[ Press Enter to speak ]")

            t_start = time.time()

            # 1. Record audio
            audio = record_until_enter()
            path  = save_wav(audio)

            # 2. IPC classification — local, fast (~50ms)
            latency_ms = (time.time() - t_start) * 1000
            ipc = extract_ipc_vector(
                path,
                prev_dominance=prev_dominance,
                filler_count=0,         # updated after transcription
                response_latency_ms=latency_ms,
            )

            # 3. Transcribe — Deepgram Nova-3 (~250ms)
            print("  Processing...")
            text, filler_count, _ = transcribe(path)
            os.unlink(path)

            # Update filler count in IPC vector
            ipc["filler_count"] = filler_count
            prev_dominance      = ipc["dominance"]

            if not text.strip():
                msg = "I didn't catch that — could you say it again?"
                print(f"  SYRA: {msg}")
                stream_tts_and_play(msg, ipc, archetype)
                turn_num -= 1  # don't count empty turns
                continue

            print(f"  Student  : {text}")
            print(f"  IPC      : dom={ipc['dominance']:.2f} | "
                  f"warm={ipc['warmth']:.2f} | "
                  f"pace={ipc['pace']} | "
                  f"fillers={filler_count} | "
                  f"giving_up={ipc['giving_up']}")

            # 4. Detect emotional state (Layer 4)
            emotional_state = detect_emotional_state(
                ipc,
                response_length=len(text),
                consecutive_confused=consecutive_confused,
            )

            # 5. Blend profile — α-weighted, session-only
            #    Does NOT change permanent memory
            adapted_ipc = pm.get_session_adapted_ipc(ipc)
            adapted_ipc["archetype"] = infer_archetype(
                adapted_ipc["dominance"],
                adapted_ipc["warmth"],
            )

            # 6. Style mirroring instruction (from previous sessions)
            mirror_instruction = get_mirror_instruction(
                style_profile, session_count
            )

            # 7. RAG retrieval — NCERT content
            rag_context = retrieve_context(text, subject, grade)

            # 8. Infer current topic (simple — last content word)
            words = [w for w in text.split() if len(w) > 3]
            if words:
                current_topic = words[-1].lower()

            # 9. Build system prompt — all 8 layers
            sys_prompt = build_system_prompt(
                adapted_ipc=adapted_ipc,
                profile=profile,
                session_mem=sm,
                subject=subject,
                grade=grade,
                rag_context=rag_context,
                turn_num=turn_num,
                current_topic=current_topic,
            )

            # Inject style mirror into prompt if available
            if mirror_instruction:
                sys_prompt = sys_prompt + f"\n\n{mirror_instruction}"

            # 10. Build messages
            conversation_history.append({
                "role":    "user",
                "content": text,
            })

            messages = [
                {"role": "system", "content": sys_prompt},
                *conversation_history[-6:],   # last 6 turns context
            ]

            # 11. Stream LLM → TTS simultaneously
            #     First sentence plays before GPT finishes generating
            reply = stream_response(
                messages=messages,
                system_prompt=sys_prompt,
                adapted_ipc=adapted_ipc,
                archetype=archetype,
            )

            conversation_history.append({
                "role":    "assistant",
                "content": reply,
            })

            # 12. Update confusion counter (Layer 6)
            was_correct = any(
                w in reply.lower()
                for w in ["exactly", "correct", "right",
                           "perfect", "well done", "yes, "]
            )
            consecutive_confused = update_confusion_counter(
                consecutive_confused, was_correct
            )

            # 13. Log turn to session memory
            sm.add_turn(text, ipc, reply, emotional_state)

            # 14. Background: update mastery map (Layer 2)
            #     Runs in thread — never blocks conversation
            if current_topic:
                threading.Thread(
                    target=_update_mastery_bg,
                    args=(text, reply, current_topic, profile),
                    daemon=True,
                ).start()

            # 15. Background: update style profile every 5 turns
            if turn_num % 5 == 0:
                def _refresh_style():
                    nonlocal style_profile
                    new    = extract_session_style(sm.turns[-10:])
                    style_profile = update_style_profile(
                        style_profile, new
                    )
                threading.Thread(
                    target=_refresh_style, daemon=True
                ).start()

            # Debug summary
            print(f"  Confused turns : {consecutive_confused}")
            print(f"  Emotional state: "
                  f"{'giving_up' if emotional_state.get('giving_up') else ''}"
                  f"{'overloaded' if emotional_state.get('overloaded') else ''}"
                  f"{'frustrated' if emotional_state.get('frustrated') else ''}"
                  f"{'normal' if not any([emotional_state.get('giving_up'),emotional_state.get('overloaded'),emotional_state.get('frustrated')]) else ''}")

    except KeyboardInterrupt:
        print("\n\nEnding session...")

    finally:
        _end_session(
            pm, sm, profile,
            subject, grade, student_id,
            style_profile,
        )


if __name__ == "__main__":
    run_session(
        subject="Mathematics",
        grade=9,
        student_id="student_001",
    )