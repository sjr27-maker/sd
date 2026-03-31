import json
from engine.llm_client import quick_extract

# Valid Mastery Levels for SYRA
MASTERY_LEVELS = ["unknown", "introduced", "practiced", "mastered"]

def update_mastery(mastery_map: dict, 
                   topic: str, 
                   new_level: str, 
                   timestamp: str) -> dict:
    """
    Updates the session's mastery map. 
    This data is eventually pushed to the permanent ProfileManager.
    """
    mastery_map[topic] = {
        "mastery": new_level,
        "last_seen": timestamp
    }
    return mastery_map

def infer_mastery_from_exchange(student_text: str, 
                                ai_response: str, 
                                topic: str) -> dict:
    """
    Analyzes the latest turn to determine if the student has moved up 
    the mastery ladder. Uses Gemini 3 Flash via quick_extract.
    """
    prompt = f"""Assess the student's mastery of "{topic}" based on this specific exchange:

Student: "{student_text}"
SYRA: "{ai_response}"

Return ONLY valid JSON:
{{
  "topic": "{topic}",
  "mastery_level": "unknown|introduced|practiced|mastered",
  "misconception_found": true|false,
  "misconception_desc": "brief description or null",
  "overconfidence": true|false
}}

Evaluation Criteria:
- mastered: Student explained correctly or solved it without SYRA's help.
- practiced: Student reached the right answer with hints/guidance.
- introduced: Student heard the concept but hasn't applied it yet.
- unknown: Student is entirely confused or lacks foundational knowledge.
- overconfidence: Student is confident in their tone but factually incorrect."""

    # Use the modular quick_extract pattern
    raw = quick_extract(prompt)

    # Standard SYRA Markdown Strip
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw.strip())
    except Exception:
        # Fallback logic to protect the session state
        return {
            "topic": topic,
            "mastery_level": "introduced",
            "misconception_found": False,
            "misconception_desc": None,
            "overconfidence": False
        }

def get_bloom_level(mastery: str) -> str:
    """
    Maps SYRA mastery levels to Bloom's Taxonomy.
    Used by the prompt_builder to adjust the difficulty of SYRA's questions.
    """
    mapping = {
        "unknown":    "remembering",
        "introduced": "understanding",
        "practiced":  "applying",
        "mastered":   "analyzing"
    }
    return mapping.get(mastery, "remembering")