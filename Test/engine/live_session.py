import asyncio
import os
import threading
import queue
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import webrtcvad

from google import genai
from google.genai import types
from dotenv import load_dotenv

from input.ipc_classifier       import extract_ipc_vector
from brain.layer1_personality   import get_ipc_style_instructions, infer_archetype
from brain.layer4_emotional     import detect_emotional_state, get_emotional_instruction
from brain.layer6_cognitive_load import get_load_instruction, update_confusion_counter
from brain.layer7_prerequisite  import get_prerequisite_instruction
from brain.layer8_comprehension import (
    should_trigger_teach_back,
    get_teach_back_instruction,
    score_session_comprehension,
)
from brain.layer2_knowledge     import infer_mastery_from_exchange
from brain.layer_style_mirror   import (
    extract_session_style,
    update_style_profile,
    get_mirror_instruction,
)
from engine.rag_retriever       import retrieve_context
from memory.profile_manager     import ProfileManager
from memory.session_memory      import SessionMemory

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("SYRA.Live")

_client    = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
LIVE_MODEL = "gemini-3.1-flash-live-preview"
IN_RATE    = 16000
OUT_RATE   = 24000
DEVICE_IDX = int(os.getenv("DEVICE_INDEX", "1"))

FRAME_SAMPLES      = 320   # exactly 20ms at 16kHz — webrtcvad requirement
END_SILENCE_FRAMES = 22    # ~440ms silence → end of student turn


# ── Streaming audio player ────────────────────────────────────────────

class StreamingPlayer:
    """
    Lock-free PCM streaming via sounddevice output callback.
    No pygame, no queue limits, no speed artifacts.
    """
    def __init__(self, rate: int = OUT_RATE):
        self._buf   = np.array([], dtype=np.int16)
        self._lock  = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            callback=self._callback,
            blocksize=2400,   # 100ms at 24kHz
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            take = min(len(self._buf), frames)
            outdata[:take, 0] = self._buf[:take] if take else np.array([], dtype=np.int16)
            self._buf = self._buf[take:]
            if take < frames:
                outdata[take:, 0] = 0

    def feed(self, raw_bytes: bytes):
        arr = np.frombuffer(raw_bytes, dtype=np.int16).copy()
        with self._lock:
            self._buf = np.concatenate([self._buf, arr])

    def stop(self):
        with self._lock:
            self._buf = np.array([], dtype=np.int16)

    @property
    def playing(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def close(self):
        self._stream.stop()
        self._stream.close()


# ── Adaptive prompt builder ───────────────────────────────────────────

def build_adaptive_prompt(
        profile:              dict,
        pm:                   ProfileManager,
        sm:                   SessionMemory,
        subject:              str,
        grade:                int,
        turn_num:             int,
        adapted_ipc:          dict,
        emotional_state:      dict,
        consecutive_confused: int,
        current_topic:        Optional[str],
        style_profile:        dict,
        session_count:        int,
        last_student_text:    str,
) -> str:
    """
    Builds a fully adaptive system prompt for each turn reconnect.
    This is called fresh on every turn — the restart architecture
    gives us free per-turn adaptation without text injection hacks.
    """

    # ── Layer 1: IPC personality ──────────────────────────────────────
    ipc_instruction = get_ipc_style_instructions(adapted_ipc)

    # ── Layer 4: Emotional state ──────────────────────────────────────
    emotional_instruction = get_emotional_instruction(emotional_state)

    # ── Layer 6: Cognitive load ───────────────────────────────────────
    chunk_size       = profile.get("cognitive_load", {}).get("chunk_size", 2)
    load_instruction = get_load_instruction(consecutive_confused, chunk_size)

    # ── Layer 7: Prerequisites ────────────────────────────────────────
    prereq_instruction = ""
    if current_topic:
        mastery_map        = profile.get("knowledge", {}).get("mastery_map", {})
        prereq_instruction = get_prerequisite_instruction(
            current_topic, mastery_map
        )

    # ── Layer 8: Teach-back ───────────────────────────────────────────
    teach_back = (
        get_teach_back_instruction()
        if should_trigger_teach_back(turn_num)
        else ""
    )

    # ── RAG context ───────────────────────────────────────────────────
    rag_context = ""
    if last_student_text and current_topic:
        rag_context = retrieve_context(last_student_text, subject, grade)

    # ── Style mirroring ───────────────────────────────────────────────
    mirror_instruction = get_mirror_instruction(style_profile, session_count)

    # ── Returning student context ─────────────────────────────────────
    ctx              = sm.context
    returning_ctx    = ""
    if ctx.get("returning") and ctx.get("last_session"):
        last = ctx["last_session"]
        rec  = last.get("recommended_next_topic", "")
        comp = last.get("comprehension_score", "?")
        returning_ctx = (
            f"\nRETURNING STUDENT: "
            f"{ctx.get('total_sessions', 0)} prior sessions. "
            f"Last comprehension: {comp}/100. "
            f"Previously struggled: {last.get('topics_struggling', [])}. "
            + (f"Pick up from: {rec}." if rec else "")
        )

    # ── Unresolved misconceptions ─────────────────────────────────────
    misconceptions = ctx.get("unresolved_misconceptions", [])
    misconception_hint = ""
    if misconceptions:
        desc = misconceptions[0].get("description", "")
        misconception_hint = (
            f"\nKNOWN MISCONCEPTION: Student previously believed '{desc}'. "
            f"If relevant, gently surface and correct it."
        )

    # ── Conversation history (last 4 turns for context) ───────────────
    history_ctx = ""
    if sm.turns:
        recent = sm.turns[-4:]
        lines  = []
        for t in recent:
            s = t.get("student_text", "")
            a = t.get("ai_response",  "")
            if s:
                lines.append(f"Student: {s}")
            if a:
                lines.append(f"SYRA: {a}")
        if lines:
            history_ctx = (
                "\nRECENT CONVERSATION:\n"
                + "\n".join(lines)
                + "\nContinue naturally from here."
            )

    # ── Goal and learning style ───────────────────────────────────────
    ls   = profile.get("learning_style", {})
    goal = profile.get("knowledge",      {}).get("goal_type", "boards")
    processing    = ls.get("processing_style",   "flexible")
    encouragement = ls.get("encouragement_need", "medium")

    # ── Assemble prompt ───────────────────────────────────────────────
    parts = [
        f"You are SYRA, an adaptive AI tutor for Class {grade} {subject}.",
        "Use NCERT-aligned content. Always end with a question or challenge.",
        "Voice output only — no markdown, bullets, or symbols.",
        "Max 70 words unless student asks for depth.",
        "",
        "INTERACTION STYLE:",
        ipc_instruction,
        "",
        f"STUDENT PROFILE:",
        f"- Goal: {goal}",
        f"- Processing: {processing} "
          f"({'overview first' if processing == 'top_down' else 'step by step'})",
        f"- Encouragement needed: {encouragement}",
        f"- Max new concepts per response: {chunk_size}",
        returning_ctx,
        misconception_hint,
        "",
        "CURRENT SESSION STATE:",
        f"- Turn: {turn_num}",
        f"- Consecutive confused turns: {consecutive_confused}",
    ]

    if emotional_instruction:
        parts += ["", f"EMOTIONAL PRIORITY:", emotional_instruction]

    parts += ["", "COGNITIVE LOAD:", load_instruction]

    if prereq_instruction:
        parts += ["", "PREREQUISITES:", prereq_instruction]

    if teach_back:
        parts += ["", "COMPREHENSION CHECK:", teach_back]

    if rag_context:
        parts += [
            "",
            f"RELEVANT NCERT CONTENT ({subject} Class {grade}):",
            rag_context[:500],
        ]

    if mirror_instruction:
        parts += ["", mirror_instruction]

    if history_ctx:
        parts += [history_ctx]

    parts += [
        "",
        "Remember: feel like a knowledgeable friend, not a formal system.",
        "Adapt your tone, pace, and vocabulary to this student specifically.",
    ]

    return "\n".join(p for p in parts if p is not None)


# ── Full duplex session ───────────────────────────────────────────────

class FullDuplexSession:

    def __init__(self,
                 profile:     dict,
                 pm:          ProfileManager,
                 session_mem: SessionMemory,
                 subject:     str,
                 grade:       int):

        self.profile     = profile
        self.pm          = pm
        self.sm          = session_mem
        self.subject     = subject
        self.grade       = grade
        self.student_id  = profile.get("student_id", "student")
        self.archetype   = profile.get("ipc", {}).get("archetype", "lina")

        # Adaptation state — persists across turns within session
        self.adapted_ipc          = {
            "dominance":           profile["ipc"].get("dominance",  0.5),
            "warmth":              profile["ipc"].get("warmth",     0.6),
            "pace":                profile["ipc"].get("pace",       "medium"),
            "giving_up":           False,
            "filler_count":        0,
            "assertiveness_delta": 0.0,
        }
        self.consecutive_confused  = 0
        self.turn_num              = 0
        self.current_topic: Optional[str] = None
        self.last_student_text     = ""
        self.prev_dominance        = profile["ipc"].get("dominance", 0.5)
        self.style_profile         = profile.get("style_profile", {})
        self.session_count         = len(profile.get("session_history", []))

        # Audio capture buffer for IPC classification
        self._turn_audio_chunks: list = []

        self._running = True
        self._player  = StreamingPlayer(OUT_RATE)

    # ── IPC classification from captured turn audio ───────────────────

    def _classify_turn_ipc(self) -> dict:
        """
        Runs IPC classification on audio captured during this student turn.
        Called after send loop ends — has full turn audio available.
        Returns updated adapted_ipc.
        """
        if not self._turn_audio_chunks:
            return self.adapted_ipc

        try:
            audio = np.concatenate(self._turn_audio_chunks).astype(np.float32)
            tmp   = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav.write(
                tmp.name, IN_RATE,
                (audio * 32767).astype(np.int16)
            )

            ipc = extract_ipc_vector(
                tmp.name,
                prev_dominance=self.prev_dominance,
                filler_count=0,
                response_latency_ms=0.0,
            )
            os.unlink(tmp.name)

            # α-blend with permanent profile — drift protected
            blended              = self.pm.get_session_adapted_ipc(ipc)
            blended["archetype"] = infer_archetype(
                blended["dominance"], blended["warmth"]
            )
            blended["assertiveness_delta"] = ipc.get(
                "assertiveness_delta", 0.0
            )

            self.prev_dominance = ipc["dominance"]
            return blended

        except Exception as e:
            logger.debug(f"IPC classify error: {e}")
            return self.adapted_ipc

    # ── Config — built fresh each turn ───────────────────────────────

    def _build_config(self) -> types.LiveConnectConfig:
        """
        System prompt rebuilt on every turn reconnect.
        This is the adaptation mechanism — each turn the model
        receives a fully updated prompt with latest IPC, emotional
        state, cognitive load, style mirroring, and conversation history.
        """
        sys_prompt = build_adaptive_prompt(
            profile              = self.profile,
            pm                   = self.pm,
            sm                   = self.sm,
            subject              = self.subject,
            grade                = self.grade,
            turn_num             = self.turn_num,
            adapted_ipc          = self.adapted_ipc,
            emotional_state      = detect_emotional_state(
                                       self.adapted_ipc,
                                       response_length=len(self.last_student_text),
                                       consecutive_confused=self.consecutive_confused,
                                   ),
            consecutive_confused = self.consecutive_confused,
            current_topic        = self.current_topic,
            style_profile        = self.style_profile,
            session_count        = self.session_count,
            last_student_text    = self.last_student_text,
        )

        voice = "Fenrir" if self.archetype == "arjun" else "Aoede"

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=sys_prompt)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

    # ── Send loop — one turn ──────────────────────────────────────────

    async def _send_loop_single_turn(self, session):
        """
        Stream mic audio for one student turn.
        Ends when 440ms silence detected after speech.
        Saves audio chunks for post-turn IPC classification.
        """
        audio_q = queue.Queue(maxsize=60)
        vad     = webrtcvad.Vad(2)
        self._turn_audio_chunks = []   # reset for this turn

        def mic_cb(indata, frames, time_info, status):
            try:
                audio_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        in_speech      = False
        silence_frames = 0

        with sd.InputStream(
            samplerate=IN_RATE,
            channels=1,
            device=DEVICE_IDX,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=mic_cb,
        ):
            while True:
                try:
                    chunk = await asyncio.to_thread(
                        audio_q.get, True, 0.5
                    )
                except queue.Empty:
                    continue

                chunk = np.clip(chunk, -1.0, 1.0)
                pcm16 = (chunk * 32767).astype(np.int16)

                # VAD
                try:
                    has_speech = vad.is_speech(pcm16.tobytes(), IN_RATE)
                except Exception:
                    has_speech = float(np.abs(chunk).mean()) > 0.02

                if has_speech:
                    silence_frames = 0
                    in_speech      = True

                    # Store for IPC classification
                    self._turn_audio_chunks.append(chunk.copy())

                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=pcm16.tobytes(),
                            mime_type="audio/pcm;rate=16000",
                        )
                    )

                else:
                    if in_speech:
                        silence_frames += 1
                        if silence_frames >= END_SILENCE_FRAMES:
                            # Student finished speaking
                            await session.send_realtime_input(
                                audio_stream_end=True
                            )
                            return

    # ── Receive loop — one turn ───────────────────────────────────────

    async def _receive_loop_single_turn(self, session) -> str:
        """
        Receive Gemini's response for one turn.
        Returns AI transcript text when turn completes.
        """
        current_txt = ""
        ai_text     = ""

        async for response in session.receive():
            sc = getattr(response, "server_content", None)
            if not sc:
                continue

            # Student transcript
            it = getattr(sc, "input_transcription", None)
            if it:
                txt = getattr(it, "text", "").strip()
                if txt:
                    print(f"\n  You: {txt}")
                    self.last_student_text = txt

                    # Infer topic from student speech
                    words = [w for w in txt.split() if len(w) > 3]
                    if words:
                        self.current_topic = words[-1].lower()

                    # Check for session end
                    if any(w in txt.lower() for w in [
                        "bye syra", "goodbye syra",
                        "stop session", "end session",
                    ]):
                        self._running = False
                        return ""

            # AI audio — feed to player
            mt = getattr(sc, "model_turn", None)
            if mt and mt.parts:
                for part in mt.parts:
                    d = getattr(part, "inline_data", None)
                    if d and getattr(d, "data", None):
                        self._player.feed(d.data)

            # AI transcript — stream to terminal
            ot = getattr(sc, "output_transcription", None)
            if ot:
                txt = getattr(ot, "text", "").strip()
                if txt and len(txt) > len(current_txt):
                    current_txt = txt
                    print(f"\r  SYRA: {current_txt}", end="", flush=True)
                    ai_text = current_txt

            # Turn complete
            if getattr(sc, "turn_complete", False):
                if current_txt:
                    print()
                return ai_text

        return ai_text

    # ── Post-turn adaptation ──────────────────────────────────────────

    def _adapt_after_turn(self, student_text: str, ai_text: str):
        """
        Runs after each turn completes.
        Updates all adaptation signals for the next turn's prompt.
        Fast enough to run synchronously between turns.
        """

        # 1. IPC classification from this turn's audio
        self.adapted_ipc = self._classify_turn_ipc()

        # 2. Emotional state
        emotional_state = detect_emotional_state(
            self.adapted_ipc,
            response_length=len(student_text),
            consecutive_confused=self.consecutive_confused,
        )

        # 3. Confusion counter
        was_correct = any(
            w in ai_text.lower()
            for w in ["exactly", "correct", "right",
                       "perfect", "well done", "yes,"]
        )
        self.consecutive_confused = update_confusion_counter(
            self.consecutive_confused, was_correct
        )

        # 4. Log turn to session memory
        self.sm.add_turn(
            student_text,
            self.adapted_ipc,
            ai_text,
            emotional_state,
        )

        # 5. Background: mastery inference
        if self.current_topic and student_text and ai_text:
            threading.Thread(
                target=self._update_mastery_bg,
                args=(student_text, ai_text, self.current_topic),
                daemon=True,
            ).start()

        # 6. Style update every 3 turns (batch from session turns)
        if self.turn_num % 3 == 0 and len(self.sm.turns) >= 3:
            threading.Thread(
                target=self._update_style_bg,
                daemon=True,
            ).start()

        # Debug log
        print(
            f"  [IPC] dom={self.adapted_ipc['dominance']:.2f} | "
            f"warm={self.adapted_ipc['warmth']:.2f} | "
            f"pace={self.adapted_ipc['pace']} | "
            f"confused={self.consecutive_confused} | "
            f"giving_up={self.adapted_ipc.get('giving_up', False)}"
        )

    def _update_mastery_bg(self, student_text: str,
                            ai_text: str, topic: str):
        try:
            result  = infer_mastery_from_exchange(student_text, ai_text, topic)
            mastery = result.get("mastery_level", "introduced")
            self.profile.setdefault(
                "knowledge", {}
            ).setdefault("mastery_map", {})[topic] = {
                "mastery":   mastery,
                "last_seen": datetime.now().isoformat(),
            }

            if result.get("misconception_found") and result.get("misconception_desc"):
                desc     = result["misconception_desc"]
                existing = self.profile.get(
                    "mental_model", {}
                ).get("naive_theories", [])
                descs = [
                    m.get("description", "") if isinstance(m, dict) else m
                    for m in existing
                ]
                if desc not in descs:
                    existing.append({
                        "description": desc,
                        "resolved":    False,
                        "found_date":  datetime.now().isoformat(),
                    })
                    self.profile.setdefault(
                        "mental_model", {}
                    )["naive_theories"] = existing

        except Exception as e:
            logger.debug(f"Mastery bg error: {e}")

    def _update_style_bg(self):
        try:
            new_style        = extract_session_style(self.sm.turns[-10:])
            self.style_profile = update_style_profile(
                self.style_profile, new_style
            )
        except Exception as e:
            logger.debug(f"Style bg error: {e}")

    # ── Wait for audio drain ──────────────────────────────────────────

    async def _wait_for_audio_drain(self):
        """Wait for player to finish before next turn starts."""
        await asyncio.sleep(0.3)
        while self._player.playing:
            await asyncio.sleep(0.08)
        await asyncio.sleep(0.15)   # small gap feels natural

    # ── Main run loop ─────────────────────────────────────────────────

    async def run(self):
        print("\n" + "="*52)
        print(f"  SYRA Live | Class {self.grade} {self.subject}")
        print(f"  Archetype: {self.archetype.upper()}")
        print("  Speak naturally. Say 'bye SYRA' to end.")
        print("="*52)

        # Turn 0 — greeting turn (no student audio, Gemini speaks first)
        print("\n  Starting session — SYRA will greet you...\n")
        try:
            greeting_config = self._build_config()
            async with _client.aio.live.connect(
                model=LIVE_MODEL,
                config=greeting_config,
            ) as session:
                # Send a silent trigger to make Gemini start
                await session.send_realtime_input(
                    text="[Session started. Greet the student warmly and ask what topic they want to cover today.]"
                )
                await self._receive_loop_single_turn(session)
            await self._wait_for_audio_drain()
        except Exception as e:
            print(f"  Greeting error: {e}")

        # Main conversation loop
        while self._running:
            self.turn_num += 1
            print(f"\n  🎤 Listening (turn {self.turn_num})...")

            try:
                async with _client.aio.live.connect(
                    model=LIVE_MODEL,
                    config=self._build_config(),   # fresh adaptive prompt
                ) as session:

                    # Run send and receive concurrently
                    _, ai_text = await asyncio.gather(
                        self._send_loop_single_turn(session),
                        self._receive_loop_single_turn(session),
                    )

                # Update all adaptation signals between turns
                self._adapt_after_turn(
                    self.last_student_text, ai_text or ""
                )

                # Wait for audio to finish before next turn
                await self._wait_for_audio_drain()

            except asyncio.TimeoutError:
                print("  Turn timeout — restarting...")
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"  Turn error: {e}")
                if not self._running:
                    break
                await asyncio.sleep(0.5)

        # Session end
        await self._end_session()

    # ── Session end ───────────────────────────────────────────────────

    async def _end_session(self):
        """Score session and update all permanent memory."""
        self._player.close()

        if not self.sm.turns:
            print("\n  No turns to save.")
            return

        print("\n  Saving session...")

        # Comprehension scoring
        try:
            session_data = score_session_comprehension(
                self.sm.turns, self.subject
            )
            print(f"  Comprehension : {session_data.get('comprehension_score', 0)}/100")
        except Exception as e:
            logger.debug(f"Scoring error: {e}")
            session_data = {
                "comprehension_score": 0,
                "topics_covered": [],
                "topics_struggling": [],
                "new_misconceptions": [],
                "recommended_next_topic": None,
            }

        # IPC summary
        dom_vals  = [t["ipc_vector"]["dominance"] for t in self.sm.turns
                     if t.get("ipc_vector")]
        warm_vals = [t["ipc_vector"]["warmth"]    for t in self.sm.turns
                     if t.get("ipc_vector")]

        ipc_summary = {
            "avg_dominance": round(
                sum(dom_vals) / len(dom_vals), 3
            ) if dom_vals else 0.5,
            "avg_warmth": round(
                sum(warm_vals) / len(warm_vals), 3
            ) if warm_vals else 0.6,
        }

        # Final style profile update
        try:
            final_style    = extract_session_style(self.sm.turns)
            self.style_profile = update_style_profile(
                self.style_profile, final_style
            )
            self.pm.profile["style_profile"] = self.style_profile
        except Exception as e:
            logger.debug(f"Style final update error: {e}")

        # Gated IPC update to permanent profile
        self.pm.update_base_profile(ipc_summary, len(self.sm.turns))

        # Knowledge update
        self.pm.update_knowledge(
            mastery_updates={},
            new_misconceptions=session_data.get("new_misconceptions", []),
            comprehension_score=session_data.get("comprehension_score", 0),
            topics_covered=session_data.get("topics_covered", []),
            topics_struggling=session_data.get("topics_struggling", []),
            recommended_next=session_data.get("recommended_next_topic"),
        )

        self.pm.save()

        # Save session log
        log = self.sm.to_log()
        log.update(session_data)
        log["ipc_summary"] = ipc_summary

        out_dir  = Path(f"sessions/{self.profile.get('student_id', 'student_001')}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.sm.to_log() | session_data | {"ipc_summary": ipc_summary},
                      f, indent=2, ensure_ascii=False)

        print(f"  Saved → {out_path}")
        print(f"  α next: {self.pm.get_alpha():.2f} | "
              f"arch: {self.pm.profile['ipc']['archetype']} | "
              f"slang: {len(self.style_profile.get('confirmed_slang', []))} confirmed")


# ── Entry point ───────────────────────────────────────────────────────

async def run_live_session(
    subject:    str = "Mathematics",
    grade:      int = 9,
    student_id: str = "student_001",
):
    from onboarding.questionnaire import run_onboarding

    pm      = ProfileManager(student_id)
    profile = run_onboarding(student_id, pm)
    sm      = SessionMemory(profile, subject)

    live = FullDuplexSession(profile, pm, sm, subject, grade)

    try:
        await live.run()
    except KeyboardInterrupt:
        print("\n  Interrupted — saving...")
        await live._end_session()


if __name__ == "__main__":
    asyncio.run(run_live_session(
        subject="Mathematics",
        grade=9,
        student_id="student_001",
    ))