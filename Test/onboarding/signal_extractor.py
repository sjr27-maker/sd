import json
from engine.llm_client import quick_extract

def extract_all_signals(qa_pairs: list[dict]) -> list[dict]:
    """
    Batch extract signals from multiple Q&A pairs.
    qa_pairs: [{"question": ..., "answer": ...}, ...]
    Returns list of signal dicts.
    """
    results = []
    for pair in qa_pairs:
        signals = extract_signals_single(
            pair["question"], pair["answer"]
        )
        results.append(signals)
    return results

def extract_signals_single(question: str, answer: str) -> dict:
    prompt = f"""Extract learning profile signals from this student answer.

Question: {question}
Answer: {answer}

Return ONLY valid JSON with these fields (null if undetectable):
{{
  "subject_difficulty": null,
  "failure_response": null,
  "processing_style": null,
  "analogy_receptiveness": null,
  "socratic_tolerance": null,
  "goal_type": null,
  "help_seeking_style": null,
  "explanation_quality": null,
  "encouragement_need": null,
  "depth_vs_breadth": null,
  "misconception_seed": null,
  "emotional_signal": null,
  "persistence_signal": null,
  "metacognition_signal": null,
  "abstraction_comfort": null
}}

Rules:
- failure_response: "retry" | "ask_someone" | "move_on" | "give_up" | null
- processing_style: "top_down" | "bottom_up" | "flexible" | null
- analogy_receptiveness: "high" | "medium" | "low" | null
- socratic_tolerance: "high" | "medium" | "low" | null
- goal_type: "boards" | "jee_neet" | "curiosity" | null
- help_seeking_style: "proactive" | "reactive" | "independent" | null
- explanation_quality: "deep" | "structured" | "surface" | null
- encouragement_need: "high" | "medium" | "low" | null
- depth_vs_breadth: "depth" | "breadth" | "balanced" | null
- emotional_signal: "anxious" | "confident" | "defeated" | "curious" | "neutral" | null
- persistence_signal: "high" | "medium" | "low" | null
- metacognition_signal: "high" | "medium" | "low" | null
- abstraction_comfort: "high" | "medium" | "low" | null"""

    raw = quick_extract(prompt)

    # Strip markdown code fences if Gemini wraps response
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw.strip())
    except Exception:
        return {}


def summarise_profile_from_signals(signals: list[dict]) -> str:
    """
    Human-readable summary of extracted profile signals.
    Useful for debugging and teacher reports.
    """
    def most_common(lst):
        cleaned = [v for v in lst if v and v != "null"]
        return max(set(cleaned), key=cleaned.count) if cleaned else "unclear"

    lines = [
        f"Processing style     : {most_common([s.get('processing_style') for s in signals])}",
        f"Goal                 : {most_common([s.get('goal_type') for s in signals])}",
        f"Failure response     : {most_common([s.get('failure_response') for s in signals])}",
        f"Encouragement need   : {most_common([s.get('encouragement_need') for s in signals])}",
        f"Emotional baseline   : {most_common([s.get('emotional_signal') for s in signals])}",
        f"Analogy receptive    : {most_common([s.get('analogy_receptiveness') for s in signals])}",
        f"Help seeking style   : {most_common([s.get('help_seeking_style') for s in signals])}",
        f"Metacognition        : {most_common([s.get('metacognition_signal') for s in signals])}",
        f"Misconceptions found : {[s.get('misconception_seed') for s in signals if s.get('misconception_seed')]}",
    ]
    return "\n".join(lines)  # this is correct