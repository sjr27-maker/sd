def get_chunk_size(processing_style: str,
                   working_memory_estimate: str) -> int:
    """
    How many new concepts per turn.
    Based on onboarding signals.
    """
    if working_memory_estimate == "low" or processing_style == "bottom_up":
        return 1
    elif working_memory_estimate == "high" or processing_style == "top_down":
        return 3
    return 2

def get_load_instruction(consecutive_confused: int,
                          chunk_size: int) -> str:
    if consecutive_confused >= 3:
        return (
            f"COGNITIVE OVERLOAD DETECTED ({consecutive_confused} confused turns). "
            "Use maximum 1 sentence. Ask yes/no only. "
            "Do not introduce any new information."
        )
    return (
        f"Introduce maximum {chunk_size} new concept(s) per response. "
        "Do not exceed this — student's working memory is calibrated to this limit."
    )

def update_confusion_counter(current_count: int,
                              was_correct: bool) -> int:
    if was_correct:
        return 0         # reset on success
    return current_count + 1