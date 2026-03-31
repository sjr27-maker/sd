import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

from input.recorder       import record_until_enter, save_wav
from input.transcriber    import transcribe
from input.ipc_classifier import extract_ipc_vector
from brain.layer2_knowledge   import infer_mastery_from_exchange
from brain.layer4_emotional   import detect_emotional_state
from brain.layer6_cognitive_load import update_confusion_counter
from brain.layer8_comprehension  import score_session_comprehension
from brain.layer1_personality    import infer_archetype
from engine.prompt_builder import build_system_prompt
from engine.llm_client      import stream_response
from engine.rag_retriever   import retrieve_context
from memory.profile_manager import ProfileManager
from memory.session_memory  import SessionMemory
from onboarding.questionnaire import run_onboarding
from output.tts_client      import stream_tts_and_play

load_dotenv()

def run_session(subject: str = "Mathematics",
                grade:   int = 9,
                student_id: str = "student_001"):

    # ── Setup ────────────────────────────────────────────────────────
    pm      = ProfileManager(student_id)
    profile = run_onboarding(student_id, pm)
    sm      = SessionMemory(profile, subject)
    archetype = profile["ipc"]["archetype"]
    
    # ACCUMULATOR: Store mastery data from every turn
    session_mastery_accumulated = {}

    print(f"\n{'='*54}")
    print(f"  SYRA | Class {grade} {subject} | {archetype.upper()}")
    print(f"  Speak → Press Enter to stop. Ctrl+C to end.")
    print(f"{'='*54}\n")

    # ── Personalised opening ─────────────────────────────────────────
    ctx = sm.context
    if ctx["returning"] and ctx.get("last_session"):
        last = ctx["last_session"]
        rec  = last.get("recommended_next_topic", "")
        opening = (
            f"Welcome back! Last time your comprehension score was "
            f"{last.get('comprehension_score', '?')}. "
            f"{'Ready to continue with ' + rec + '?' if rec else 'What shall we work on today?'}"
        )
    else:
        openings = {
            "maya":  f"Let's go through {subject} together. What would you like to start with?",
            "arjun": f"Let's get into {subject}. What are we working on?",
            "lina":  f"Let's get started with {subject}. Which topic today?",
        }
        opening = openings.get(archetype, openings["lina"])

    seed_ipc = {
        "dominance": profile["ipc"]["dominance"],
        "warmth":    profile["ipc"]["warmth"],
        "pace":      profile["ipc"]["pace"],
        "giving_up": False,
    }
    print(f"  SYRA: {opening}")
    stream_tts_and_play(opening, seed_ipc, archetype)

    conversation_history = []
    turn_num             = 0
    prev_dominance       = profile["ipc"]["dominance"]

    # ── Session loop ─────────────────────────────────────────────────
    try:
        while True:
            turn_num += 1
            print(f"\n── Turn {turn_num} {'─'*40}")
            input("[ Press Enter to speak ]")

            t_start = time.time()

            # 1. Record
            audio = record_until_enter()
            path  = save_wav(audio)

            # 2. Transcribe FIRST (to get accurate filler counts)
            text, filler_count, _ = transcribe(path)
            
            # 3. IPC classify (Pass the filler_count we just found)
            latency_ms = (time.time() - t_start) * 1000
            ipc = extract_ipc_vector(
                path,
                prev_dominance=prev_dominance,
                filler_count=filler_count,
                response_latency_ms=latency_ms
            )
            os.unlink(path) # Clean up temp wav

            if not text:
                stream_tts_and_play(
                    "I didn't catch that — could you say it again?",
                    ipc, archetype
                )
                continue

            # Update state
            ipc["filler_count"] = filler_count
            prev_dominance      = ipc["dominance"]

            print(f"  Student : {text}")
            print(f"  IPC     : dom={ipc['dominance']} | warm={ipc['warmth']} | pace={ipc['pace']}")

            # 4. Emotional state detection
            emotional_state = detect_emotional_state(
                ipc,
                response_length=len(text),
                consecutive_confused=sm.consecutive_confused
            )

            # 5. Blend profile
            adapted_ipc = pm.get_session_adapted_ipc(ipc)
            adapted_ipc["archetype"] = infer_archetype(
                adapted_ipc["dominance"], adapted_ipc["warmth"]
            )

            # 6. RAG retrieval
            rag_context = retrieve_context(text, subject, grade)

            # 7. Build prompt
            sys_prompt = build_system_prompt(
                adapted_ipc=adapted_ipc,
                profile=profile,
                session_mem=sm,
                subject=subject,
                grade=grade,
                rag_context=rag_context,
                turn_num=turn_num,
                current_topic=sm.current_topic
            )

            conversation_history.append({"role": "user", "content": text})
            messages = [*conversation_history[-6:]]

            # 8. THE FIX: Pass 4 arguments (including system_prompt)
            reply = stream_response(messages, sys_prompt, adapted_ipc, archetype)
            
            conversation_history.append({"role": "assistant", "content": reply})

            # 9. Update knowledge map
            if sm.current_topic:
                mastery_result = infer_mastery_from_exchange(text, reply, sm.current_topic)
                # Store in our accumulator for end-of-session save
                session_mastery_accumulated[sm.current_topic] = mastery_result.get("mastery_level", "introduced")

            # 10. Update confusion counter
            was_correct = ("correct" in reply.lower() or "exactly" in reply.lower() or "yes" in reply.lower()[:20])
            sm.consecutive_confused = update_confusion_counter(sm.consecutive_confused, was_correct)

            # 11. Log turn
            sm.add_turn(text, ipc, reply, emotional_state)

    except KeyboardInterrupt:
        print("\n\nEnding session...")
        _end_session(pm, sm, profile, subject, grade, student_id, session_mastery_accumulated)

def _end_session(pm, sm, profile, subject, grade, student_id, mastery_updates):
    """Score session and update permanent memory."""

    # Score comprehension
    session_data = score_session_comprehension(sm.turns, subject)
    
    # IPC summary
    if sm.turns:
        dom_vals  = [t["ipc_vector"]["dominance"] for t in sm.turns]
        warm_vals = [t["ipc_vector"]["warmth"]    for t in sm.turns]
        ipc_summary = {
            "avg_dominance":    round(sum(dom_vals)/len(dom_vals), 3),
            "avg_warmth":       round(sum(warm_vals)/len(warm_vals), 3),
            "avg_filler_count": round(sum(t["ipc_vector"].get("filler_count", 0) for t in sm.turns) / len(sm.turns), 1),
        }
    else:
        ipc_summary = {}

    # Update permanent memory
    pm.update_base_profile(ipc_summary, len(sm.turns))

    # Update knowledge (using the actual mastery_updates we accumulated)
    pm.update_knowledge(
        mastery_updates=mastery_updates, 
        new_misconceptions=session_data.get("new_misconceptions", []),
        comprehension_score=session_data.get("comprehension_score", 0),
        topics_covered=session_data.get("topics_covered", []),
        topics_struggling=session_data.get("topics_struggling", []),
        recommended_next=session_data.get("recommended_next_topic"),
    )

    # Save session log
    session_log = sm.to_log()
    session_log.update(session_data)
    session_log["ipc_summary"] = ipc_summary

    path = f"sessions/{student_id}/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(f"sessions/{student_id}", exist_ok=True)
    with open(path, "w") as f:
        json.dump(session_log, f, indent=2)

    print(f"\n  Session saved → {path}")
    print(f"  PM updated. α next session: {pm.get_alpha()}")