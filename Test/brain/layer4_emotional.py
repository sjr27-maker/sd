def detect_emotional_state(ipc: dict,
                             response_length: int,
                             consecutive_confused: int) -> dict:
    """
    Detect emotional state from IPC signals and response patterns.
    Returns emotional state dict.
    """
    state = {
        "giving_up":        ipc.get("giving_up", False),
        "enthusiastic":     ipc.get("enthusiasm", False),
        "frustrated":       False,
        "anxious":          False,
        "recovering":       False,
    }

    # Frustration: short answers + high fillers + low warmth
    if (response_length < 15
            and ipc.get("filler_count", 0) > 3
            and ipc.get("warmth", 0.5) < 0.35):
        state["frustrated"] = True

    # Anxiety: high filler count + volume drop + slow pace
    if (ipc.get("filler_count", 0) > 5
            and ipc.get("volume_drop", False)
            and ipc.get("pace") == "slow"):
        state["anxious"] = True

    # Cognitive overload
    state["overloaded"] = consecutive_confused >= 3

    return state

def get_emotional_instruction(state: dict) -> str:
    """Returns prompt instruction based on emotional state."""
    if state.get("giving_up"):
        return (
            "Student is giving up. Maximum warmth. One simple question. "
            "Praise any attempt. Create safety before continuing."
        )
    if state.get("overloaded"):
        return (
            "Student is cognitively overloaded. Simplify drastically. "
            "One sentence at a time. Ask only yes/no questions."
        )
    if state.get("frustrated"):
        return (
            "Student is frustrated. Acknowledge the difficulty first: "
            "'This is actually a tricky one.' Then reframe with a simpler angle."
        )
    if state.get("anxious"):
        return (
            "Student seems anxious. Slow down. Be very reassuring. "
            "Remind them there's no rush and mistakes are expected."
        )
    if state.get("enthusiastic"):
        return (
            "Student is enthusiastic and engaged. Match their energy. "
            "Build on the momentum — go slightly deeper."
        )
    return ""  # Normal state — no override needed