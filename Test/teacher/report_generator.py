import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# SYRA Standard Imports
from engine.llm_client import quick_extract

# Paths & Constants
SESSIONS_DIR = Path("sessions")
REPORTS_DIR  = Path("reports")
logger = logging.getLogger("SYRA.Reporter")

# ── Data Loaders ──────────────────────────────────────────────────────

def _load_profile(student_id: str) -> Dict[str, Any]:
    """Safely loads the student's long-term profile."""
    path = SESSIONS_DIR / student_id / "student_profile.json"
    if not path.exists(): return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except: return {}

def _load_sessions(student_id: str) -> List[Dict[str, Any]]:
    """Gathers all session JSONs sorted by date."""
    folder = SESSIONS_DIR / student_id
    if not folder.exists(): return []
    sessions = []
    for f in sorted(folder.glob("session_*.json")):
        try:
            sessions.append(json.loads(f.read_text(encoding="utf-8")))
        except: continue
    return sessions

# ── Metric Helpers ───────────────────────────────────────────────────

def _avg(values: list) -> float:
    """Calculates average, skipping None values."""
    cleaned = [v for v in values if v is not None]
    return round(sum(cleaned) / len(cleaned), 1) if cleaned else 0.0

def _calculate_trend(values: list) -> str:
    """Determines if the student's performance is improving or stable."""
    if len(values) < 2: return "insufficient data"
    # Using a 10% delta for 'improving/declining' threshold
    delta = values[-1] - values[0]
    if delta > 10: return "improving"
    if delta < -10: return "declining"
    return "stable"

# ── Report Generation ───────────────────────────────────────────────

def _generate_narrative(report_data: Dict[str, Any]) -> str:
    """Uses Gemini 3 Flash to write a professional narrative summary for the teacher."""
    
    # Prune the data to avoid context-window bloat
    summary_for_ai = {
        "archetype": report_data["student_overview"]["archetype"],
        "goal": report_data["student_overview"]["goal"],
        "avg_score": report_data["academic_progress"]["avg_comprehension_score"],
        "struggles": report_data["academic_progress"]["top_struggle_topics"],
        "unresolved_misconceptions": report_data["misconceptions"]["unresolved"],
        "frustration_rate": report_data["emotional_patterns"]["frustration_rate"]
    }

    prompt = f"""You are a senior academic advisor writing a teacher report summary.
Student Data: {json.dumps(summary_for_ai)}

Write a professional 3-sentence summary for a classroom teacher.
1. Highlight the student's current learning archetype and overall progress.
2. Identify the single most critical misconception or struggle topic.
3. Provide one high-impact pedagogical suggestion for the next lesson.

Tone: Professional, empathetic, and actionable."""

    # Replaces the old _client.chat.completions call
    return quick_extract(prompt).strip()

def build_report(student_id: str) -> Dict[str, Any]:
    """Orchestrates data aggregation and AI-driven narrative synthesis."""
    profile = _load_profile(student_id)
    sessions = _load_sessions(student_id)

    if not profile:
        return {"error": f"No data found for {student_id}"}

    # Extract Performance Values
    comp_scores = [s.get("comprehension_score") for s in sessions if s.get("comprehension_score")]
    dom_values = [s.get("ipc_summary", {}).get("avg_dominance") for s in sessions if s.get("ipc_summary")]

    # Aggregate Misconceptions
    unresolved = [m.get("description") for m in profile.get("mental_model", {}).get("naive_theories", []) if not m.get("resolved")]

    # Identify Topic Struggles
    all_struggles = []
    for s in sessions: all_struggles.extend(s.get("topics_struggling", []))
    struggle_freq = {t: all_struggles.count(t) for t in set(all_struggles)}
    top_struggles = sorted(struggle_freq.items(), key=lambda x: x[1], reverse=True)[:3]

    report = {
        "student_id": student_id,
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "student_overview": {
            "archetype": profile.get("ipc", {}).get("archetype", "lina"),
            "goal": profile.get("knowledge", {}).get("goal_type", "curiosity"),
            "total_sessions": len(sessions)
        },
        "academic_progress": {
            "avg_comprehension_score": _avg(comp_scores),
            "comprehension_trend": _calculate_trend(comp_scores),
            "top_struggle_topics": [t[0] for t in top_struggles]
        },
        "misconceptions": {
            "unresolved": unresolved,
            "total_found": len(profile.get("knowledge", {}).get("misconceptions", []))
        },
        "emotional_patterns": {
            "frustration_rate": _avg([s.get("ipc_summary", {}).get("avg_warmth", 0) for s in sessions]) # Simplified
        }
    }

    # Add the AI Layer
    report["ai_narrative"] = _generate_narrative(report)
    return report

# ── Main Entry Point ──────────────────────────────────────────────────

def generate_report(student_id: str):
    """Main function to build, save, and display the report."""
    REPORTS_DIR.mkdir(exist_ok=True)
    
    print(f"📊 Generating SYRA Report for {student_id}...")
    report = build_report(student_id)

    if "error" in report:
        print(f"❌ {report['error']}")
        return

    # Save to disk
    report_path = REPORTS_DIR / f"{student_id}_report.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    # Console Summary
    print("\n" + "="*56)
    print(f" SYRA TEACHER REPORT: {student_id.upper()}")
    print("="*56)
    print(f" Archetype  : {report['student_overview']['archetype'].upper()}")
    print(f" Mastery    : {report['academic_progress']['avg_comprehension_score']}% ({report['academic_progress']['comprehension_trend']})")
    print(f"\n Advisor Narrative:")
    print(f" {report['ai_narrative']}")
    print("\n" + "="*56)
    
    return report