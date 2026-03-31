import json
from engine.llm_client import quick_extract

# --- Constants ---
ERROR_TYPES = ["careless", "conceptual", "prerequisite_gap", "language_confusion"]

# --- Teach-Back Logic ---

def should_trigger_teach_back(turn_num: int) -> bool:
    """Trigger teach-back every 3rd turn to verify student understanding."""
    return turn_num > 0 and turn_num % 3 == 0

def get_teach_back_instruction() -> str:
    """Returns the pedagogical instruction for the prompt builder."""
    return (
        "TEACH-BACK MODE: Ask the student to explain what they just learned "
        "back to you in their own words. Use a prompt like: 'To make sure I've explained "
        "that clearly, how would you describe this concept to a friend?' "
        "Listen for core keywords and logical gaps."
    )

# --- Error Analysis ---

def classify_error(student_text: str, correct_answer: str, topic: str) -> str:
    """
    Classifies a student's mistake to determine if SYRA should 
    provide a hint, a correction, or back-track to prerequisites.
    """
    prompt = f"""Student gave a wrong answer regarding "{topic}".
Student response: "{student_text}"
Correct reference: "{correct_answer}"

Classify the error type as exactly ONE of these words:
- careless: Small slip, the student clearly knows the concept.
- conceptual: Fundamental misunderstanding of the current topic.
- prerequisite_gap: Missing foundational knowledge from earlier chapters.
- language_confusion: Understood the math/logic but expressed it poorly.

Return ONLY the single word."""

    # Using quick_extract for low-latency classification
    result = quick_extract(prompt).strip().lower()
    
    return result if result in ERROR_TYPES else "conceptual"

# --- Session End Analysis ---

def score_session_comprehension(session_turns: list, subject: str) -> dict:
    """
    Analyzes the full session transcript to update the long-term student profile.
    Uses Gemini to extract mastery signals and misconceptions.
    """
    # Format the last 12 turns for context window efficiency
    transcript = "\n".join([
        f"Student: {t.get('student_text', '')}\nSYRA: {t.get('ai_response', '')}"
        for t in session_turns[-12:]
    ])

    prompt = f"""Analyze this tutoring session transcript for {subject} and provide a summary.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON:
{{
  "comprehension_score": 0-100,
  "topics_covered": ["topic1", "topic2"],
  "topics_struggling": ["topic1"],
  "topics_mastered": ["topic2"],
  "recommended_next_topic": "topic_name",
  "new_misconceptions": ["detailed description of the misunderstanding"],
  "session_quality": "productive|mixed|struggling"
}}"""

    raw = quick_extract(prompt)

    # Strip markdown code fences (Standard SYRA parsing)
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw.strip())
    except Exception:
        # Robust fallback to prevent profile corruption
        return {
            "comprehension_score": 50,
            "topics_covered": [],
            "topics_struggling": [],
            "topics_mastered": [],
            "recommended_next_topic": None,
            "new_misconceptions": [],
            "session_quality": "mixed"
        }