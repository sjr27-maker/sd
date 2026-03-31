from brain.layer1_personality import get_ipc_style_instructions
from brain.layer4_emotional   import get_emotional_instruction
from brain.layer6_cognitive_load import get_load_instruction
from brain.layer7_prerequisite   import get_prerequisite_instruction
from brain.layer8_comprehension  import (
    should_trigger_teach_back,
    get_teach_back_instruction
)

def build_system_prompt(
        adapted_ipc:    dict,
        profile:        dict,
        session_mem,          # SessionMemory instance
        subject:        str,
        grade:          int,
        rag_context:    str,
        turn_num:       int,
        current_topic:  str = None) -> str:

    parts = []

    # 1. Core identity
    parts.append(
        f"You are an adaptive AI tutor for Class {grade} {subject}. "
        f"You teach using NCERT-aligned content. Always end responses with "
        f"either a question or a challenge — never a full stop into silence."
    )

    # 2. IPC personality instruction (Layer 1)
    ipc_instruction = get_ipc_style_instructions(adapted_ipc)
    parts.append(f"INTERACTION STYLE:\n{ipc_instruction}")

    # 3. Emotional override (Layer 4)
    emotion_instruction = get_emotional_instruction(
        session_mem.emotion_timeline[-1]
        if session_mem.emotion_timeline else {}
    )
    if emotion_instruction:
        parts.append(f"EMOTIONAL PRIORITY:\n{emotion_instruction}")

    # 4. Cognitive load (Layer 6)
    chunk_size = profile.get("cognitive_load", {}).get("chunk_size", 2)
    load_instruction = get_load_instruction(
        session_mem.consecutive_confused, chunk_size
    )
    parts.append(f"COGNITIVE LOAD:\n{load_instruction}")

    # 5. Prerequisite gaps (Layer 7)
    if current_topic:
        mastery_map = profile.get("knowledge", {}).get("mastery_map", {})
        prereq_instruction = get_prerequisite_instruction(
            current_topic, mastery_map
        )
        if prereq_instruction:
            parts.append(f"PREREQUISITES:\n{prereq_instruction}")

    # 6. Teach-back trigger (Layer 8)
    if should_trigger_teach_back(turn_num):
        parts.append(
            f"COMPREHENSION CHECK:\n{get_teach_back_instruction()}"
        )

    # 7. Returning student context
    ctx = session_mem.context
    if ctx.get("returning") and ctx.get("last_session"):
        last = ctx["last_session"]
        parts.append(
            f"STUDENT HISTORY: This student has had "
            f"{ctx['total_sessions']} sessions. "
            f"Last session comprehension: "
            f"{last.get('comprehension_score', 'unknown')}. "
            f"Previously struggled with: "
            f"{last.get('topics_struggling', [])}."
        )

    # 8. Unresolved misconceptions
    misconceptions = ctx.get("unresolved_misconceptions", [])
    if misconceptions:
        desc = misconceptions[0].get("description", "")
        parts.append(
            f"KNOWN MISCONCEPTION: Student previously believed '{desc}'. "
            f"If this topic arises, gently probe and correct."
        )

    # 9. RAG context
    if rag_context:
        parts.append(
            f"REFERENCE CONTENT (NCERT):\n{rag_context}\n"
            f"Use this content as your knowledge source."
        )

    # 10. Response format
    goal = profile.get("knowledge", {}).get("goal_type", "boards")
    arch = adapted_ipc.get("archetype", "lina")
    max_words = {"maya": 60, "arjun": 50, "lina": 70}.get(arch, 65)

    parts.append(
        f"RESPONSE FORMAT: Max {max_words} words. "
        f"Student goal: {goal}. "
        f"Calibrate difficulty accordingly."
    )

    return "\n\n".join(parts)