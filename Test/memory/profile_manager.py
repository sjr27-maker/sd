import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup basic logging for profile events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SYRA.ProfileManager")

class ProfileManager:
    """
    Manages long-term student data, including IPC personality traits,
    knowledge mastery, and learning style preferences.
    
    Implements Alpha-Blending for personality updates and 1-gate drift 
    protection to maintain profile stability.
    """

    def __init__(self, student_id: str, base_dir: str = "sessions"):
        self.student_id = student_id
        self.profile_path = Path(base_dir) / student_id / "student_profile.json"
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Hyperparameters for profile evolution
        self.alpha = 0.15  # Learning rate for personality (15% new, 85% old)
        self.drift_threshold = 0.25  # Max allowed change per session
        
        self.data = self._load_profile()

    def _load_profile(self) -> Dict[str, Any]:
        """Loads profile from disk or initializes a default if missing."""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Profile for {self.student_id} corrupted. Resetting.")
        
        return self._get_default_profile()

    def _get_default_profile(self) -> Dict[str, Any]:
        """Returns a baseline profile for new students."""
        return {
            "student_id": self.student_id,
            "ipc": {
                "dominance": 0.5,
                "warmth": 0.5,
                "pace": 1.0,
                "archetype": "lina"
            },
            "mastery_map": {},
            "misconceptions": [],
            "session_count": 0,
            "metrics": {
                "avg_comprehension": 0.0,
                "total_turns": 0
            }
        }

    def get_session_adapted_ipc(self, current_ipc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Blends the long-term profile with the immediate 'vibe' of the student
        at the start of a session to create an adapted persona.
        """
        base = self.data["ipc"]
        return {
            "dominance": round((base["dominance"] + current_ipc["dominance"]) / 2, 3),
            "warmth": round((base["warmth"] + current_ipc["warmth"]) / 2, 3),
            "pace": current_ipc.get("pace", base["pace"])
        }

    def update_base_profile(self, session_summary: Dict[str, float], turn_count: int):
        """
        Updates the long-term IPC traits using Alpha-Blending and Drift Protection.
        
        Formula: new_val = (1 - α) * old_val + α * session_val
        """
        if not session_summary:
            return

        old_ipc = self.data["ipc"]
        new_ipc = {}

        # 1-Gate Drift Protection & Alpha Blending
        for trait in ["dominance", "warmth"]:
            old_val = old_ipc.get(trait, 0.5)
            session_val = session_summary.get(f"avg_{trait}", old_val)

            # Calculate blended value
            blended = ((1 - self.alpha) * old_val) + (self.alpha * session_val)
            
            # Drift Protection: Cap the change to prevent extreme shifts
            delta = blended - old_val
            if abs(delta) > self.drift_threshold:
                direction = 1 if delta > 0 else -1
                blended = old_val + (self.drift_threshold * direction)
                logger.warning(f"Drift protection triggered for {trait} in {self.student_id}")

            new_ipc[trait] = round(blended, 3)

        # Update stats
        self.data["ipc"].update(new_ipc)
        self.data["session_count"] += 1
        self.data["metrics"]["total_turns"] += turn_count
        
        self._save()

    def update_knowledge(self, 
                         mastery_updates: Dict[str, str], 
                         new_misconceptions: List[str], 
                         **kwargs):
        """Updates the student's mastery levels and logs new areas of confusion."""
        # Update Mastery Map
        for topic, level in mastery_updates.items():
            self.data["mastery_map"][topic] = {
                "mastery": level,
                "last_updated": kwargs.get("timestamp")
            }

        # Append unique misconceptions
        current_misc = set(self.data["misconceptions"])
        current_misc.update(new_misconceptions)
        self.data["misconceptions"] = list(current_misc)

        self._save()

    def get_alpha(self) -> float:
        """Returns the current learning rate for reporting."""
        return self.alpha

    def _save(self):
        """Persists the profile to a JSON file."""
        with open(self.profile_path, "w") as f:
            json.dump(self.data, f, indent=4)
        logger.info(f"Profile saved for {self.student_id}")