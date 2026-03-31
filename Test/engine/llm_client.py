import os
from google import genai
from google.genai import types
from output.tts_client import stream_tts_and_play
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL   = "gemini-2.5-flash"   # change to gemini-2.5-flash-lite for cheaper

SENTENCE_ENDS = {".", "!", "?"}

def stream_response(messages: list,
                    system_prompt: str,
                    adapted_ipc: dict,
                    archetype: str) -> str:
    """
    Stream Gemini 2.5 Flash response with sentence-level TTS pipeline.
    Thinking disabled for lowest latency.
    Returns full response string.
    """
    # Convert OpenAI-style message list to Gemini contents format
    # Gemini doesn't have "system" role in contents —
    # system instruction is a separate parameter
    contents = []
    for msg in messages:
        role = "user" if msg["role"] in ("user", "system") else "model"
        if msg["role"] == "system":
            continue  # handled via system_instruction below
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            )
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=130,
        temperature=0.72,
        thinking_config=types.ThinkingConfig(
            thinking_budget=0   # CRITICAL — disable thinking for speed
        ),
    )

    full_reply   = ""
    sentence_buf = ""

    response_stream = _client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=config,
    )

    for chunk in response_stream:
        if not chunk.text:
            continue

        delta         = chunk.text
        full_reply   += delta
        sentence_buf += delta

        # Sentence boundary → play immediately
        stripped = sentence_buf.rstrip()
        if stripped and stripped[-1] in SENTENCE_ENDS:
            sentence = sentence_buf.strip()
            if sentence:
                print(f"  SYRA: {sentence}")
                stream_tts_and_play(sentence, adapted_ipc, archetype)
            sentence_buf = ""

    # Flush remaining
    if sentence_buf.strip():
        print(f"  SYRA: {sentence_buf.strip()}")
        stream_tts_and_play(sentence_buf.strip(), adapted_ipc, archetype)

    return full_reply.strip()


def quick_extract(prompt: str) -> str:
    """
    Non-streaming call for background tasks:
    knowledge map updates, comprehension scoring,
    onboarding signal extraction, error classification.
    Uses flash-lite for cost efficiency on these tasks.
    """
    response = _client.models.generate_content(
        model="gemini-2.5-flash-lite",  # cheaper for analysis tasks
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=400,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()