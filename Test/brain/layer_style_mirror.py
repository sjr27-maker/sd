# brain/layer_style_mirror.py

from engine.llm_client import quick_extract
import json

STYLE_EXTRACT_PROMPT = """Analyse this student's single spoken turn.

Student said: "{text}"

Extract ONLY valid JSON:
{{
  "vocabulary_level": "simple|casual|academic|mixed",
  "slang_detected": ["list", "of", "slang", "words", "or", "phrases"],
  "humor_style": "none|self_deprecating|observational|sarcastic|playful",
  "sentence_length": "short|medium|long",
  "formality": "very_informal|informal|neutral",
  "filler_style": ["like", "you know", "basically", "literally"],
  "enthusiasm_markers": ["bro", "man", "dude", "yaar", "da"],
  "pace_words": "fast_talker|medium|deliberate",
  "positive_signals": ["any encouraging phrases student used"],
  "example_phrase": "one short phrase that captures their style"
}}

Be precise. Only include slang/fillers that actually appeared."""


def extract_style_fingerprint(student_text: str) -> dict:
    """Extract one turn's style signals."""
    raw = quick_extract(
        STYLE_EXTRACT_PROMPT.format(text=student_text)
    )
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {}


def update_style_profile(existing: dict, new_turn: dict) -> dict:
    """
    Rolling update of the student's style profile.
    Uses frequency counting — a slang word must appear in
    3+ turns before it's considered part of their style.
    Never adds negative or harmful slang.
    """
    if not new_turn:
        return existing

    profile = existing.copy()

    # Vocabulary level — most common across last 10 turns
    levels = profile.get("vocab_history", [])
    if new_turn.get("vocabulary_level"):
        levels.append(new_turn["vocabulary_level"])
        profile["vocab_history"] = levels[-10:]  # keep last 10
        profile["vocabulary_level"] = _most_common(levels[-10:])

    # Slang accumulation — frequency gated
    slang_counts = profile.get("slang_counts", {})
    for word in new_turn.get("slang_detected", []):
        word = word.lower().strip()
        if word and len(word) > 1:
            slang_counts[word] = slang_counts.get(word, 0) + 1
    profile["slang_counts"] = slang_counts
    # Only use slang seen 3+ times — prevents single-use noise
    profile["confirmed_slang"] = [
        w for w, c in slang_counts.items() if c >= 3
    ]

    # Humor style — only update if consistent
    humor_history = profile.get("humor_history", [])
    if new_turn.get("humor_style") and new_turn["humor_style"] != "none":
        humor_history.append(new_turn["humor_style"])
        profile["humor_history"] = humor_history[-8:]
        profile["humor_style"] = _most_common(humor_history[-8:])

    # Enthusiasm markers — these are gold
    markers = profile.get("enthusiasm_markers", set())
    if isinstance(markers, list):
        markers = set(markers)
    for m in new_turn.get("enthusiasm_markers", []):
        markers.add(m.lower().strip())
    profile["enthusiasm_markers"] = list(markers)

    # Formality — most common
    form_history = profile.get("formality_history", [])
    if new_turn.get("formality"):
        form_history.append(new_turn["formality"])
        profile["formality_history"] = form_history[-10:]
        profile["formality"] = _most_common(form_history[-10:])

    # Pace
    if new_turn.get("pace_words"):
        profile["pace_words"] = new_turn["pace_words"]

    # Sentence length preference
    len_history = profile.get("length_history", [])
    if new_turn.get("sentence_length"):
        len_history.append(new_turn["sentence_length"])
        profile["length_history"] = len_history[-10:]
        profile["sentence_length"] = _most_common(len_history[-10:])

    # Example phrases — keep last 5 unique ones
    examples = profile.get("example_phrases", [])
    if new_turn.get("example_phrase"):
        examples.append(new_turn["example_phrase"])
        profile["example_phrases"] = list(dict.fromkeys(examples))[-5:]

    # Session count
    profile["turns_analysed"] = profile.get("turns_analysed", 0) + 1

    return profile


def _most_common(lst: list):
    if not lst:
        return None
    return max(set(lst), key=lst.count)


def get_mirror_instruction(style_profile: dict,
                            session_count: int) -> str:
    """
    Generate the style mirroring instruction for the prompt.
    Gradually increases mirroring as sessions accumulate.
    Session 1-2: very subtle. Session 5+: natural friend-like.
    """
    if not style_profile or style_profile.get("turns_analysed", 0) < 3:
        return ""  # not enough data yet

    turns    = style_profile.get("turns_analysed", 0)
    slang    = style_profile.get("confirmed_slang", [])
    markers  = style_profile.get("enthusiasm_markers", [])
    humor    = style_profile.get("humor_style", "none")
    formality = style_profile.get("formality", "neutral")
    vocab    = style_profile.get("vocabulary_level", "casual")
    examples = style_profile.get("example_phrases", [])

    # Gate: only start mirroring after enough turns
    if turns < 5:
        intensity = "very subtle"
        slang_instruction = ""
    elif turns < 15:
        intensity = "gentle"
        slang_instruction = (
            f"Occasionally use these words the student uses naturally: "
            f"{', '.join(slang[:3])}. "
            if slang else ""
        )
    else:
        intensity = "natural"
        slang_instruction = (
            f"Naturally use words from their vocabulary: "
            f"{', '.join(slang[:5])}. "
            if slang else ""
        )

    # Formality instruction
    if formality in ("very_informal", "informal"):
        formality_instruction = (
            "Keep your language casual and relaxed. "
            "No stiff or overly formal phrasing. "
        )
    else:
        formality_instruction = ""

    # Humor instruction — only if student shows it
    if humor == "playful" and session_count >= 2:
        humor_instruction = (
            "If the moment naturally allows, a light playful comment "
            "or joke is welcome — like a friend would. "
            "Never force it. Only when it fits. "
        )
    elif humor == "self_deprecating" and session_count >= 3:
        humor_instruction = (
            "Occasionally use light self-aware humor — "
            "the student appreciates that style. "
        )
    elif humor == "sarcastic" and session_count >= 3:
        humor_instruction = (
            "The student uses dry/sarcastic humor. "
            "You can occasionally match that tone — lightly, "
            "never at the student's expense. "
        )
    else:
        humor_instruction = ""

    # Enthusiasm markers — yaar, bro, dude etc
    marker_instruction = ""
    if markers and turns >= 10:
        safe_markers = [
            m for m in markers
            if m in {"yaar", "bro", "man", "dude", "da",
                     "buddy", "mate", "boss"}
        ]
        if safe_markers:
            marker_instruction = (
                f"You can occasionally use '{safe_markers[0]}' "
                f"like they do — only when it feels natural. "
            )

    # Sentence length
    length_map = {
        "short":  "Keep your responses punchy and short — like they talk.",
        "medium": "Match their natural conversational pace.",
        "long":   "You can give slightly longer explanations — they engage with detail.",
    }
    length_instruction = length_map.get(
        style_profile.get("sentence_length", "medium"), ""
    )

    # Example phrases context
    example_instruction = ""
    if examples and turns >= 20:
        example_instruction = (
            f"Their natural speech sounds like: "
            f"'{examples[-1]}'. Mirror that register. "
        )

    parts = [
        p for p in [
            f"STYLE MIRRORING ({intensity}):",
            formality_instruction,
            slang_instruction,
            marker_instruction,
            humor_instruction,
            length_instruction,
            example_instruction,
            "This should feel like talking to a knowledgeable friend, "
            "not a formal tutor. Natural, warm, real.",
        ] if p
    ]

    return "\n".join(parts)

    # Add these two functions to your existing layer_style_mirror.py

def extract_session_style(turns: list) -> dict:
    """
    Called ONCE at session end.
    Analyses all turns together — more accurate than per-turn.
    """
    from engine.llm_client import quick_extract
    import json

    if len(turns) < 3:
        return {}

    transcript = "\n".join([
        f"Student: {t['student_text']}"
        for t in turns[-12:]
        if t.get("student_text")
    ])

    prompt = f"""Analyse this student's speech across the session.

{transcript}

Return ONLY valid JSON:
{{
  "vocabulary_level": "simple|casual|academic|mixed",
  "confirmed_slang": ["words used repeatedly — only real slang"],
  "humor_style": "none|playful|sarcastic|self_deprecating",
  "formality": "very_informal|informal|neutral",
  "sentence_length": "short|medium|long",
  "enthusiasm_markers": ["yaar", "bro", "etc — only if actually used"],
  "example_phrases": ["up to 3 phrases capturing their style"],
  "humorCount": 0
}}

Only include slang confirmed across multiple turns. Be conservative."""

    raw = quick_extract(prompt)
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {}


def update_style_profile(existing: dict, new_session: dict) -> dict:
    """
    Merge new session style data into existing profile.
    Frequency-gated — slang must appear across multiple sessions.
    """
    if not new_session:
        return existing

    profile = existing.copy()

    # Slang frequency counting across sessions
    counts = profile.get("slang_counts", {})
    for word in new_session.get("confirmed_slang", []):
        word = word.lower().strip()
        if word:
            counts[word] = counts.get(word, 0) + 1
    profile["slang_counts"] = counts
    # Confirmed after appearing in 2+ sessions
    profile["confirmed_slang"] = [
        w for w, c in counts.items() if c >= 2
    ][:8]

    # Humor count accumulation
    profile["humorCount"] = (
        profile.get("humorCount", 0)
        + new_session.get("humorCount", 0)
    )

    # Most common formality across sessions
    form_hist = profile.get("formality_history", [])
    if new_session.get("formality"):
        form_hist.append(new_session["formality"])
        profile["formality_history"] = form_hist[-8:]
        vals = [v for v in form_hist if v]
        profile["formality"] = max(set(vals), key=vals.count) if vals else "neutral"

    # Enthusiasm markers accumulate
    markers = set(profile.get("enthusiasm_markers", []))
    for m in new_session.get("enthusiasm_markers", []):
        markers.add(m.lower().strip())
    profile["enthusiasm_markers"] = list(markers)

    # Example phrases — keep freshest
    examples = profile.get("example_phrases", [])
    examples.extend(new_session.get("example_phrases", []))
    profile["example_phrases"] = list(dict.fromkeys(examples))[-5:]

    # Sessions analysed count
    profile["sessions_analysed"] = profile.get("sessions_analysed", 0) + 1

    return profile