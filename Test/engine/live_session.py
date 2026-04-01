import asyncio
import os
import numpy as np
import sounddevice as sd
import pygame
from google import genai
from google.genai import types
from input.ipc_classifier import extract_ipc_vector
from memory.profile_manager import ProfileManager
from memory.session_memory import SessionMemory
from engine.prompt_builder import build_system_prompt
from dotenv import load_dotenv
import tempfile
import scipy.io.wavfile as wav

load_dotenv()

# Global Client setup
_client      = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
LIVE_MODEL   = "gemini-3.1-flash-live-preview"
SAMPLE_RATE  = 16000
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "1"))
CHUNK_FRAMES = 1600    # 100ms chunks at 16kHz

# Initialize Pygame Mixer for 24kHz (Gemini's native output)
# 1. Start with 48kHz (High Fidelity) and a safe buffer of 4096 to stop breaking
pygame.mixer.pre_init(frequency=48000, size=-16, channels=1, buffer=4096)
pygame.mixer.init()

# 2. Check what frequency the OS actually gave us
ACTUAL_FREQ, _, ACTUAL_CHANS = pygame.mixer.get_init()
print(f"--- Audio Hardware: {ACTUAL_FREQ}Hz | {'Stereo' if ACTUAL_CHANS==2 else 'Mono'} ---")

class FullDuplexSession:
    """
    SYRA Full-Duplex Voice Session.
    Enables natural conversation with real-time interruption and IPC adaptation.
    """

    def __init__(self, profile: dict, pm: ProfileManager,
                 session_mem: SessionMemory,
                 subject: str, grade: int):
        self.profile     = profile
        self.pm          = pm
        self.sm          = session_mem
        self.subject     = subject
        self.grade       = grade
        self.archetype   = profile["ipc"]["archetype"]
        self.adapted_ipc = {
            "dominance":  profile["ipc"]["dominance"],
            "warmth":     profile["ipc"]["warmth"],
            "pace":       profile["ipc"]["pace"],
            "giving_up":  False,
        }

        self._audio_buffer   = []
        self._ipc_lock       = asyncio.Lock()
        self._running        = True
        self._turn_count     = 0

    def _build_live_config(self) -> types.LiveConnectConfig:
        """Build the configuration for Gemini 3.1 Flash Live."""
        sys_prompt = build_system_prompt(
            adapted_ipc=self.adapted_ipc,
            profile=self.profile,
            session_mem=self.sm,
            subject=self.subject,
            grade=self.grade,
            rag_context="",
            turn_num=self._turn_count,
        )

        return types.LiveConnectConfig(
            # Using only AUDIO prevents 1011 Internal Errors in the Preview version
            response_modalities=["AUDIO"],
            
            system_instruction=types.Content(parts=[types.Part(text=sys_prompt)]),
            
            # Using a dictionary ensures compatibility if HistoryConfig naming changes
            history_config={
                "initial_history_in_client_content": True
            },
            
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede" if self.archetype in ("maya", "lina") else "Fenrir"
                    )
                )
            )
        )

    async def _classify_ipc_periodically(self):
        """Analyze student tone every 2 seconds to adapt AI personality."""
        while self._running:
            await asyncio.sleep(2.0)
            if len(self._audio_buffer) < SAMPLE_RATE * 2:
                continue

            audio_chunk = np.concatenate(self._audio_buffer[-SAMPLE_RATE * 2:]).astype(np.float32)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav.write(tmp.name, SAMPLE_RATE, (audio_chunk * 32767).astype(np.int16))

            try:
                ipc = extract_ipc_vector(tmp.name, prev_dominance=self.adapted_ipc.get("dominance", 0.5))
                async with self._ipc_lock:
                    self.adapted_ipc = self.pm.get_session_adapted_ipc(ipc)
                    self.adapted_ipc["archetype"] = self.archetype

                print(f"  [IPC] dom={self.adapted_ipc['dominance']} | warm={self.adapted_ipc['warmth']}")
            except Exception as e:
                print(f"  [IPC error: {e}]")
            finally:
                os.unlink(tmp.name)

            if len(self._audio_buffer) > SAMPLE_RATE * 10:
                self._audio_buffer = self._audio_buffer[-SAMPLE_RATE * 10:]

    async def run(self):
        """Main connection and loop handler."""
        print(f"\n{'='*54}")
        print(f"  SYRA Live | {self.subject} | {self.archetype.upper()}")
        print(f"  Speak naturally. Say 'goodbye' to end.")
        print(f"{'='*54}\n")

        config = self._build_live_config()

        try:
            async with _client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                ipc_task = asyncio.create_task(self._classify_ipc_periodically())

                # Send seeding content
                opening = f"Let's get started with {self.subject}. What topic should we cover?"
                await session.send_client_content(
                    turns=[types.Content(role="user", parts=[types.Part(text=f"[Start] {opening}")])],
                    turn_complete=True
                )

                # Run Mic and AI Audio in parallel
                await asyncio.gather(
                    self._send_audio(session),
                    self._receive_audio(session),
                )
        except Exception as e:
            print(f"\n[Connection Error] {e}")
        finally:
            self._running = False
            ipc_task.cancel()
            pygame.mixer.stop()

    async def _send_audio(self, session):
        audio_queue = asyncio.Queue()

        def mic_callback(indata, frames, time_info, status):
            audio_queue.put_nowait(indata.copy().flatten())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                          device=DEVICE_INDEX, blocksize=CHUNK_FRAMES, callback=mic_callback):
            while self._running:
                chunk = await audio_queue.get()
                self._audio_buffer.append(chunk)
                pcm_bytes = (chunk * 32767).astype(np.int16).tobytes()
                await session.send_realtime_input(audio=types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000"))

    async def _receive_audio(self, session):
        ai_channel = pygame.mixer.Channel(0)

        async for response in session.receive():
            if not self._running: break
            
            # 1. Safely get server content
            sc = getattr(response, "server_content", None)
            if not sc: continue

            # 2. Safely check for Student (Input) Transcription
            it = getattr(sc, "input_transcription", None)
            if it and getattr(it, "text", None):
                student_text = it.text.strip()
                if student_text:
                    ai_channel.stop() # Interruption logic
                    print(f"  Student: {student_text}")
                    if "goodbye" in student_text.lower():
                        self._running = False
                        return

            # 3. Safely check for AI (Output) Transcription (Optional logging)
            ot = getattr(sc, "output_transcription", None)
            if ot and getattr(ot, "text", None):
                print(f"  SYRA: {ot.text}")

            # 4. Handle AI Audio
            mt = getattr(sc, "model_turn", None)
            if mt and mt.parts:
                for part in mt.parts:
                    data = getattr(part, "inline_data", None)
                    if data and data.data:
                        audio_data = np.frombuffer(data.data, dtype=np.int16)
                        if audio_data.size > 0:
                            # Standard 24k/48k/Stereo fixes
                            mixer_config = pygame.mixer.get_init()
                            if mixer_config[0] == 48000: audio_data = np.repeat(audio_data, 2)
                            if mixer_config[2] == 2: audio_data = np.column_stack((audio_data, audio_data))

                            sound = pygame.sndarray.make_sound(audio_data)
                            ai_channel.play(sound)
                            while ai_channel.get_busy() and self._running:
                                await asyncio.sleep(0.002)
                                
async def run_live_session(subject="Mathematics", grade=9, student_id="student_001"):
    from onboarding.questionnaire import run_onboarding
    pm = ProfileManager(student_id)
    profile = run_onboarding(student_id, pm)
    sm = SessionMemory(profile, subject)

    session = FullDuplexSession(profile, pm, sm, subject, grade)
    await session.run()

    # Session Wrap-up
    from brain.layer8_comprehension import score_session_comprehension
    if sm.turns:
        session_data = score_session_comprehension(sm.turns, subject)
        print(f"\n  Session complete. Score: {session_data.get('comprehension_score')}/100")
        pm.save()

if __name__ == "__main__":
    asyncio.run(run_live_session(subject="Mathematics", grade=9, student_id="student_001"))