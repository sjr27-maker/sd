import json
import os

_GRAPH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "prerequisite_graph.json"
)

def load_graph() -> dict:
    if os.path.exists(_GRAPH_PATH):
        with open(_GRAPH_PATH) as f:
            return json.load(f)
    # Hardcoded fallback — Class 9 Maths
    return {
        "quadratic_equations":  ["linear_equations", "factorisation"],
        "trigonometry":         ["pythagoras_theorem", "similar_triangles"],
        "coordinate_geometry":  ["number_line", "linear_equations"],
        "polynomials":          ["algebraic_expressions", "factorisation"],
        "surface_area_volume":  ["mensuration_2d", "basic_geometry"],
        "statistics":           ["data_representation", "arithmetic_mean"],
        "probability":          ["statistics", "fractions"],
        "circles":              ["basic_geometry", "angles"],
        "triangles":            ["basic_geometry", "congruence"],
        "number_systems":       [],   # root topic
        "linear_equations":     ["number_systems"],
        "factorisation":        ["algebraic_expressions"],
    }

_GRAPH = load_graph()

def get_prerequisites(topic: str) -> list[str]:
    return _GRAPH.get(topic.lower().replace(" ", "_"), [])

def find_unmastered_prerequisites(
        topic: str, mastery_map: dict) -> list[str]:
    """
    Returns list of prerequisites for topic that aren't mastered yet.
    These are the root cause gaps.
    """
    prereqs  = get_prerequisites(topic)
    unmastered = []
    for p in prereqs:
        status = mastery_map.get(p, {})
        level  = status.get("mastery", "unknown") if isinstance(
            status, dict) else "unknown"
        if level in ["unknown", "introduced"]:
            unmastered.append(p)
    return unmastered

def get_prerequisite_instruction(topic: str,
                                  mastery_map: dict) -> str:
    gaps = find_unmastered_prerequisites(topic, mastery_map)
    if not gaps:
        return ""
    gap_list = ", ".join(gaps)
    return (
        f"WARNING: Student has prerequisite gaps in: {gap_list}. "
        f"Before continuing with {topic}, briefly check their understanding "
        f"of {gaps[0]} first. Do not skip prerequisites."
    )