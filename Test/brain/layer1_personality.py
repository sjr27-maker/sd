def infer_archetype(dominance: float, warmth: float) -> str:
    if dominance < 0.35 and warmth > 0.50:
        return "maya"    # shy, warm, curious
    elif dominance > 0.65:
        return "arjun"   # confident, assertive
    else:
        return "lina"    # balanced, needs clarity

def compute_persistence_signal(
        prev_turn_correct: bool,
        student_tried_again: bool) -> str:
    """Infer persistence from whether student retried after wrong answer."""
    if prev_turn_correct:
        return "not_applicable"
    return "high" if student_tried_again else "low"

def get_ipc_style_instructions(adapted_ipc: dict) -> str:
    """
    Returns natural language instructions for prompt_builder
    based on adapted IPC profile.
    """
    dom      = adapted_ipc["dominance"]
    warm     = adapted_ipc["warmth"]
    pace     = adapted_ipc["pace"]
    giving_up = adapted_ipc.get("giving_up", False)
    arch     = infer_archetype(dom, warm)

    # Giving-up always overrides everything
    if giving_up:
        return (
            "URGENT: Student appears to be giving up or disengaging. "
            "Immediately switch to maximum warmth. Ask one very simple "
            "yes/no question. Praise any attempt. Do not continue the "
            "current topic — create a safe moment first."
        )

    if arch == "maya":
        base = (
            "Student is shy and hesitant. Be warm and deeply encouraging. "
            "Ask only ONE question at a time — never stack questions. "
            "Celebrate every correct step explicitly. When wrong, guide with "
            "'What do you think happens if...' rather than correcting directly. "
            "Use simple analogies. Never sound impatient."
        )
    elif arch == "arjun":
        base = (
            "Student is confident and fast-paced. Be direct and concise. "
            "Skip unnecessary praise — they find it patronising. "
            "After every correct answer, immediately raise difficulty. "
            "If wrong, be direct: 'Not quite — think about this part.' "
            "Match their energy. They respect competence over warmth."
        )
    else:
        base = (
            "Student needs clarity and patience. Use real-world examples "
            "before abstract concepts. Check understanding frequently: "
            "'Does that make sense so far?' Medium warmth — friendly but focused. "
            "One concept at a time."
        )

    # Pace modifier
    if pace == "slow":
        base += " Student is speaking slowly — keep responses brief and patient."
    elif pace == "fast":
        base += " Student is engaged and fast — keep their energy up."

    return base