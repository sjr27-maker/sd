from datetime import datetime

class SessionMemory:
    """Short-term memory for current session only."""

    def __init__(self, profile: dict, subject: str):
        self.subject              = subject
        self.turns                = []
        self.emotion_timeline     = []
        self.consecutive_confused = 0
        self.teach_back_counter   = 0
        self.current_topic        = None
        self.giving_up_flag       = False
        self.prev_dominance       = profile.get("ipc", {}).get("dominance", 0.5)
        self.start_time           = datetime.now()

        # Derive relevant context from permanent memory
        self.context = self._derive_from_pm(profile, subject)

    def _derive_from_pm(self, profile: dict, subject: str) -> dict:
        """Pull only relevant context for this session from PM."""
        history = profile.get("session_history", [])
        mastery = profile.get("knowledge", {}).get("mastery_map", {})
        now     = datetime.now()

        # Topics due for spaced review (not seen in 7+ days)
        due_review = []
        for topic, data in mastery.items():
            if isinstance(data, dict) and data.get("last_seen"):
                try:
                    days = (now - datetime.fromisoformat(
                        data["last_seen"])).days
                    if days > 7 and data.get("mastery") in [
                        "practiced", "mastered"
                    ]:
                        due_review.append(topic)
                except Exception:
                    pass

        # Unresolved misconceptions
        unresolved = [
            m for m in profile.get(
                "mental_model", {}
            ).get("naive_theories", [])
            if isinstance(m, dict) and not m.get("resolved", False)
        ]

        # Last session summary
        last = history[-1] if history else None

        return {
            "returning":            len(history) > 0,
            "last_session":         last,
            "topics_due_review":    due_review,
            "unresolved_misconceptions": unresolved,
            "total_sessions":       len(history),
        }

    def add_turn(self, student_text: str, ipc: dict,
                 ai_response: str, emotional_state: dict):
        self.turns.append({
            "turn":          len(self.turns) + 1,
            "timestamp":     datetime.now().isoformat(),
            "student_text":  student_text,
            "ipc_vector":    ipc,
            "ai_response":   ai_response,
            "emotional":     emotional_state,
            "ft_input":      {"ipc": ipc, "student": student_text},
            "ft_target":     ai_response,
        })
        self.emotion_timeline.append(emotional_state)
        self.prev_dominance = ipc.get("dominance", self.prev_dominance)
        if emotional_state.get("giving_up"):
            self.giving_up_flag = True

    def update_confusion(self, was_correct: bool):
        if was_correct:
            self.consecutive_confused = 0
        else:
            self.consecutive_confused += 1

    def to_log(self) -> dict:
        return {
            "subject":            self.subject,
            "start_time":         self.start_time.isoformat(),
            "end_time":           datetime.now().isoformat(),
            "turn_count":         len(self.turns),
            "turns":              self.turns,
            "emotion_timeline":   self.emotion_timeline,
        }