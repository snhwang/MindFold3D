"""
LLM-Mediated Cognitive Coaching Layer for MindFold 3D.

Copyright (c) 2024-2026 Scott N. Hwang, Parviz Safadel. All rights reserved.
Patent pending. See docs/PATENT_SPECIFICATION.md for claims.

This module implements the LLM-Mediated Cognitive Coaching System, a core
novel component of the MindFold 3D invention (Claims 5, 15, 18).

Key inventive methods:
  - serialize_performance_for_llm():   Structured performance data serialization (Claim 5c)
  - build_quick_feedback_prompt():     Post-response diagnostic coaching (Claim 5d)
  - build_session_summary_prompt():    Cross-dimensional cognitive synthesis (Claim 5d)
  - build_scorecard_analysis_prompt(): Nine-skill taxonomy mapping (Claim 5b)
  - get_fallback_coaching():           Rule-based deterministic fallback (Claim 5f)

Uses an OpenAI-compatible API (Ollama, LM Studio, vLLM, etc.) to provide
personalized cognitive coaching based on spatial reasoning performance data.
"""

import os
import time
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

_client = None


def get_llm_client():
    """Lazy-initialize the OpenAI client. Returns None if unavailable."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI(
                base_url=LLM_BASE_URL,
                api_key=os.getenv("LLM_API_KEY", "not-needed"),
                timeout=LLM_TIMEOUT,
            )
        except ImportError:
            return None
    return _client


def check_llm_available() -> bool:
    """Ping the LLM endpoint to verify availability."""
    client = get_llm_client()
    if client is None:
        return False
    try:
        client.models.list()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# System Prompt: encodes the cognitive framework from the paper (Section 7.4)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a cognitive coach for MindFold 3D, a spatial reasoning assessment and training application. You analyze performance data from a 3D shape-matching task where users identify a target voxel shape among distractors.

## Cognitive Framework

The system measures performance across a three-layer architecture:

### Layer 1: Shape Geometry (what the shape IS — optimized by the generator)
- **Spatial Form**: anisotropy_index (0=balanced, 1=elongated), shape_form_index (-1=rod, +1=pancake), planarity_score (high=flat, low=truly 3D)
- **Structural Complexity**: branching_factor (branch points in the shape graph), number_of_components (disconnected clusters), cycle_count (holes/loops in the adjacency graph; 0=tree-like, 1+=has loops)
- **Spatial Density**: compactness_score (neighbor density, 0=sparse, 1=dense), voxel_count (total cubes)

Emergent features (computed but not targeted): surface_area, bounding_box_ratio, dominant_axis, largest_component_ratio

### Layer 2: Task Design (what the user must DO)
- **Mental Rotation**: Shapes rotated in 3D; difficulty scales with angular disparity. RT increases linearly with angle.
- **Mirror Discrimination**: Mirror-reflected distractors require detecting handedness/chirality. Neurally distinct from rotation.
- **Visuospatial Working Memory**: Delayed matching (target disappears before choices appear). Capacity ~4 items (Cowan 2001).

### Layer 3: Behavioral Metrics (HOW the user responds)
- **Processing Efficiency**: Response time patterns, RT slopes across difficulty
- **Interaction Strategy**: Exploration and rotation patterns

## Task Variants
- **simultaneous**: Target and choices visible at once (low WM demand)
- **delayed-short**: Target shown 5s then removed, choices appear (moderate WM demand)
- **delayed-long**: Target shown 3s, 2s gap, then choices (high WM demand)
- **mirror mode**: Mirror reflections included as distractors
- **part-permuted mode**: Distractors with same parts rearranged at branch points (tests configural binding)
- **perspective mode**: Target shown from one canonical viewpoint, choices from a different viewpoint. Orbit controls disabled. Tests viewpoint inference (Skill 8), separable from mental rotation.

## Structural Archetypes
- **Gestalt-Encodable**: Compact, symmetric shapes (compactness>0.7, branching 0-1). Errors indicate transformation deficits, not encoding failures.
- **Analytically-Decomposable**: Branching, multi-segment shapes (branching>=3). Errors may indicate incomplete encoding rather than transformation failure.
- **Topologically-Complex**: Shapes with holes, bridges, concavities (cycle_count >= 1). Errors indicate volumetric representation deficits (Skill 3).

## Nine Target Skills
1. Holistic Shape Recognition — fast gestalt encoding of compact shapes
2. Structural Decomposition — parsing branching shapes into parts (Palmer 1977)
3. Topological Reasoning — sensitivity to holes, connectivity, genus (Chen 2005)
4. Mental Rotation — speed/accuracy of 3D rotation (Shepard & Metzler 1971)
5. Mirror Discrimination — chirality detection, distinct RT pattern from rotation (Corballis 1988)
6. Spatial Visualization — complex multi-step mental manipulation (Hegarty & Waller 2004)
7. Configural Binding — encoding spatial arrangement of parts (Hummel & Biederman 1992)
8. Perspective Taking — viewpoint inference (Kozhevnikov & Hegarty 2001)
9. Spatial Working Memory — maintenance across delay; chunking reduces load (Luck & Vogel 1997)

## Diagnostic Signatures
- Errors on branching but not compact shapes → structural decomposition deficit (Skill 2)
- Errors specifically on mirror distractors → chirality processing deficit (Skill 5)
- Performance drops with delay (delayed vs simultaneous) → working memory bottleneck (Skill 9)
- RT increases steeply with voxel count → encoding capacity limit
- Errors on high-anisotropy shapes → difficulty processing orientation-dependent views (Skill 1/4)
- Success on easy shapes but failure on complex ones at same rotation → encoding deficit, not rotation deficit
- Errors on shapes with cycle_count>0 but not on acyclic shapes → topological reasoning deficit (Skill 3): struggles with holes/loops requiring volumetric encoding
- Errors specifically on part-permuted distractors → configural binding deficit (Skill 7): encodes WHAT parts exist but not HOW they are arranged
- Errors in perspective mode but not in standard rotation mode → perspective-taking deficit (Skill 8): can rotate objects but cannot infer appearance from a different viewpoint

### Interaction-Based Diagnostics (Layer 3)
- RT increases linearly with angular disparity + minimal physical rotation → strong mental rotation (Skill 4), replicating Shepard & Metzler (1971)
- Flat RT across angular disparity + extensive physical rotation → relying on orbit controls instead of mental rotation
- High accuracy with mental_rotation strategy → efficient spatial transformer
- Low accuracy with physical_rotation strategy → over-relying on controls without effective comparison
- Interaction concentrated on target (not choices) → encoding difficulty (Skill 1/2)
- Interaction concentrated on choices → discrimination difficulty (Skill 5/7)
- Zero rotation events with correct answer → expert mental rotation or trivially easy shape
- High interaction time with correct answer → successful but effortful; may benefit from chunking

## Coaching Strategies (grounded in cognitive science)
- **Chunking**: Identify the main spine first, encode branches as offsets. Reduces WM load (Alvarez & Cavanagh 2004).
- **Hierarchical encoding**: Parse from global form to local detail (Palmer 1977).
- **Reference axis strategy**: Lock onto the dominant axis, then compare deviations.
- **Elimination strategy**: Reject obviously wrong choices first to reduce comparison load.
- **Symmetry exploitation**: Use symmetry axes as anchoring landmarks for comparison.
- **Part-relation encoding**: For branching shapes, encode not just which parts exist but WHERE each arm extends from the branch point. Think of it as "syntax" (arrangement) not just "vocabulary" (parts).
- **Volumetric scanning**: For shapes with holes/loops (cycle_count>0), mentally trace around each hole to verify it exists in both shapes. Think of the shape as a 3D volume with tunnels, not just a collection of blocks.
- **Viewpoint simulation**: For perspective tasks, imagine physically walking around the shape to the new viewpoint. Identify landmark features (branches, corners) visible from the target view, then predict which would be visible or hidden from the choice viewpoint.
- **Systematic comparison**: Before rotating shapes with orbit controls, compare the overall silhouette from the current view. Only use controls to verify specific features, not to fully align shapes. Reducing physical rotation builds mental rotation skill.

## Your Role
- Provide cross-dimensional diagnostic synthesis (not just listing weak features)
- Ground strategy recommendations in cognitive science
- Track progress narratively when historical data is available
- Be concise, specific, and encouraging
- Use the skill names and archetype vocabulary consistently
- Never fabricate performance numbers; only reference data provided
- Format with markdown bold (**text**) for emphasis"""


# ---------------------------------------------------------------------------
# Performance data serialization
# ---------------------------------------------------------------------------

def serialize_performance_for_llm(
    scorecard_data: Dict[str, Any],
    max_features: int = 15,
) -> str:
    """Convert scorecard data into a compact text summary for LLM context."""
    lines = []

    session = scorecard_data.get("session", {})
    cumulative = scorecard_data.get("cumulative", {})

    # Overall stats
    lines.append("## Current Session")
    lines.append(
        f"- Questions: {session.get('total_questions', 0)}, "
        f"Accuracy: {session.get('accuracy', 0)}%"
    )

    if cumulative.get("total_questions", 0) > 0:
        lines.append("## Cumulative (All Sessions)")
        lines.append(
            f"- Questions: {cumulative.get('total_questions', 0)}, "
            f"Accuracy: {cumulative.get('accuracy', 0)}%"
        )

    # Task variant stats
    task_stats = session.get("task_variant_stats", {})
    if task_stats:
        lines.append("## Task Variant Performance")
        for variant, stats in task_stats.items():
            lines.append(
                f"- {variant}: {stats.get('accuracy', 0)}% "
                f"({stats.get('total', 0)} trials)"
            )

    # Weak areas
    for label, data_source in [("Session", session), ("Cumulative", cumulative)]:
        weak = data_source.get("weak_areas", [])
        if weak:
            lines.append(f"## {label} Weak Areas (<50% success, >=3 attempts)")
            for area in weak:
                lines.append(
                    f"- {area['feature']}={area['value']}: "
                    f"{area['success_rate']}% ({area['attempts']} attempts)"
                )

    # Per-feature breakdown (prioritize features with data)
    for label, data_source in [("Session", session), ("Cumulative", cumulative)]:
        feature_scorecard = data_source.get("feature_scorecard", {})
        feature_summaries = []

        for feature, values in feature_scorecard.items():
            if not values:
                continue
            for value, stats in values.items():
                total = stats.get("total_attempts", 0)
                if total >= 2:
                    feature_summaries.append({
                        "feature": feature,
                        "value": value,
                        "success_rate": stats.get("success_rate", 0),
                        "attempts": total,
                        "avg_rt": stats.get("avg_response_time"),
                    })

        feature_summaries.sort(key=lambda x: (-x["attempts"], x["success_rate"]))

        if feature_summaries:
            lines.append(f"## {label} Feature Details (top {max_features})")
            for fs in feature_summaries[:max_features]:
                rt_str = f", avg RT: {fs['avg_rt']:.1f}s" if fs["avg_rt"] else ""
                lines.append(
                    f"- {fs['feature']}={fs['value']}: "
                    f"{fs['success_rate']}% ({fs['attempts']} attempts{rt_str})"
                )

    # Interaction strategy metrics (Layer 3)
    for label, data_source in [("Session", session), ("Cumulative", cumulative)]:
        interaction = data_source.get("interaction_scorecard", {})
        has_data = any(bool(v) for v in interaction.values())
        if has_data:
            lines.append(f"## {label} Interaction Strategy (Layer 3)")

            ad_stats = interaction.get("angular_disparity_bucket", {})
            if ad_stats:
                lines.append("### Angular Disparity vs Performance")
                for bucket in sorted(ad_stats.keys()):
                    stats = ad_stats[bucket]
                    rt_str = f", avg RT: {stats['avg_response_time']:.1f}s" if stats.get("avg_response_time") else ""
                    lines.append(
                        f"- {bucket} deg: {stats.get('success_rate', 0)}% accuracy "
                        f"({stats.get('total_attempts', 0)} trials{rt_str})"
                    )

            is_stats = interaction.get("interaction_strategy", {})
            if is_stats:
                lines.append("### Rotation Strategy (Mental vs Physical)")
                for strategy in sorted(is_stats.keys()):
                    stats = is_stats[strategy]
                    rt_str = f", avg RT: {stats['avg_response_time']:.1f}s" if stats.get("avg_response_time") else ""
                    lines.append(
                        f"- {strategy}: {stats.get('success_rate', 0)}% accuracy "
                        f"({stats.get('total_attempts', 0)} trials{rt_str})"
                    )

    # Recent raw attempts for cross-dimensional analysis
    recent = session.get("recent_attempts_full", [])
    if recent:
        lines.append("## Recent Attempts (cross-dimensional context)")
        for i, a in enumerate(recent[-10:], 1):
            parts = [
                "correct" if a.get("correct") else "incorrect",
                f"RT:{a.get('response_time', 0):.1f}s",
                a.get("task_variant", "simultaneous"),
            ]
            if a.get("mirror_mode"):
                parts.append("mirror")
            ad = a.get("angular_disparity")
            if ad is not None:
                parts.append(f"disparity:{ad:.0f}deg")
            rot = a.get("rotation_events", 0)
            if rot > 0:
                parts.append(f"rotations:{rot}")
            lines.append(f"- Trial {i}: {', '.join(parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders for each trigger point
# ---------------------------------------------------------------------------

def build_quick_feedback_prompt(
    was_correct: bool,
    response_time: float,
    target_features: Dict[str, Any],
    recent_attempts: List[Dict],
    session_stats_summary: str,
    interaction_data: Optional[Dict] = None,
) -> str:
    """After each response: brief, targeted feedback (1-3 sentences)."""
    feature_highlights = []
    for key in [
        "voxel_count", "branching_factor", "compactness_score",
        "planarity_score", "anisotropy_index", "shape_form_index",
        "number_of_components", "cycle_count",
    ]:
        val = target_features.get(key)
        if val is not None:
            feature_highlights.append(f"{key}={val}")

    recent_correct = sum(1 for a in recent_attempts[-5:] if a.get("correct"))
    recent_total = min(5, len(recent_attempts))

    result = "correct" if was_correct else "incorrect"

    # Build interaction context if available
    interaction_context = ""
    if interaction_data and isinstance(interaction_data, dict):
        summary = interaction_data.get("summary", {})
        angular_disparity = interaction_data.get("angular_disparity")
        total_rotation = summary.get("total_angular_displacement", 0)
        interaction_time = summary.get("total_interaction_time_ms", 0)
        parts = []
        if angular_disparity is not None:
            parts.append(f"angular disparity: {angular_disparity:.0f} deg")
        if total_rotation > 0:
            parts.append(f"physical rotation: {total_rotation:.0f} deg")
        else:
            parts.append("no physical rotation used")
        if interaction_time > 0:
            parts.append(f"interaction time: {interaction_time/1000:.1f}s")
        interaction_context = f"Interaction: {', '.join(parts)}\n"

    return (
        f"The user just answered {result} (response time: {response_time:.1f}s).\n"
        f"Shape features: {', '.join(feature_highlights)}\n"
        f"{interaction_context}"
        f"Recent performance: {recent_correct}/{recent_total} correct in last "
        f"{recent_total} trials.\n\n"
        f"{session_stats_summary}\n\n"
        "Provide 1-3 sentences of brief, targeted feedback. "
        "If incorrect, identify what made this shape challenging based on its features. "
        "If correct but slow (>8s), suggest an efficiency strategy. "
        "If the user used extensive physical rotation, consider suggesting mental rotation practice. "
        "If correct and fast with no rotation, offer brief encouragement. "
        "Do NOT repeat the performance numbers back."
    )


def build_session_summary_prompt(performance_summary: str) -> str:
    """End of session: comprehensive analysis."""
    return (
        "The user has finished a training session. "
        "Here is their complete performance data:\n\n"
        f"{performance_summary}\n\n"
        "Provide a comprehensive session summary with these sections:\n"
        "1. **Session Overview** (2-3 sentences on overall performance)\n"
        "2. **Strengths** (which cognitive skills or shape types they handle well)\n"
        "3. **Areas for Growth** (cross-dimensional diagnostic synthesis — "
        "identify the underlying cognitive bottleneck, not just weak features)\n"
        "4. **Strategy Recommendations** (1-2 specific, actionable strategies "
        "grounded in spatial cognition research)\n"
        "5. **Next Session Focus** (what to practice and why)\n\n"
        "Keep the total response under 250 words."
    )


def build_on_demand_advice_prompt(
    performance_summary: str,
    user_question: Optional[str] = None,
) -> str:
    """User-initiated coaching via button press."""
    base = (
        "The user is requesting coaching advice during their training session.\n\n"
        f"{performance_summary}\n\n"
    )
    if user_question:
        base += (
            f'The user asks: "{user_question}"\n\n'
            "Answer their specific question using the performance data "
            "and cognitive framework. "
        )
    else:
        base += "Provide personalized coaching based on the performance data. "

    base += (
        "Include:\n"
        "1. A diagnostic interpretation of their current performance pattern\n"
        "2. One specific strategy recommendation they can apply immediately\n"
        "3. A brief encouraging note about their progress\n\n"
        "Keep the total response under 150 words."
    )
    return base


def build_scorecard_analysis_prompt(performance_summary: str) -> str:
    """Detailed analysis for the scorecard page."""
    return (
        "The user is viewing their detailed performance scorecard. "
        "Provide an in-depth cognitive analysis.\n\n"
        f"{performance_summary}\n\n"
        "Provide a detailed analysis with these sections:\n"
        "1. **Cognitive Profile** (map performance patterns to the nine-skill "
        "taxonomy; classify which skills are strong vs. developing)\n"
        "2. **Cross-Dimensional Interactions** (identify where performance on "
        "one dimension affects another — e.g., 'branching shapes are harder "
        "for you specifically when mirror distractors are present')\n"
        "3. **Structural Archetype Analysis** (how well does the user handle "
        "Gestalt-Encodable vs. Analytically-Decomposable shapes?)\n"
        "4. **Temporal Patterns** (if delayed matching data exists, analyze "
        "working memory effects)\n"
        "5. **Recommended Training Sequence** (specific progression of "
        "difficulty settings and task variants)\n\n"
        "Keep the total response under 400 words."
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_coaching_response(
    user_prompt: str,
    max_tokens: int = None,
    temperature: float = 0.7,
) -> Optional[str]:
    """Send a coaching request to the LLM. Returns None on failure."""
    client = get_llm_client()
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens or LLM_MAX_TOKENS,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM coaching error: {e}")
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback when LLM is unavailable
# ---------------------------------------------------------------------------

FALLBACK_MESSAGES = {
    "branching_factor": (
        "Shapes with more branches require parsing the structure into parts. "
        "Try identifying the main 'spine' first, then encode branches as offsets."
    ),
    "compactness_score": (
        "Dense, compact shapes can be harder to distinguish. "
        "Focus on the overall silhouette first, then check specific details."
    ),
    "shape_form_index": (
        "Rod-like and disc-like shapes look very different depending on viewing angle. "
        "Identify the overall form first — is it elongated, flat, or compact?"
    ),
    "planarity_score": (
        "Truly 3D shapes (low planarity) require mental rotation in depth. "
        "Try to identify the dominant axis as your reference point."
    ),
    "voxel_count": (
        "Shapes with more voxels are harder to encode all at once. "
        "Try chunking: group nearby voxels into sub-shapes and compare chunks."
    ),
    "anisotropy_index": (
        "Elongated shapes look very different from different angles. "
        "Identify the long axis as your reference, then compare how branches "
        "deviate from it."
    ),
    "number_of_components": (
        "Multi-component shapes (disconnected pieces) require encoding each "
        "piece separately. Compare the pieces one at a time."
    ),
    "cycle_count": (
        "Shapes with holes or loops require tracking the 3D volume, not just "
        "individual blocks. Try mentally tracing around any holes to verify "
        "both shapes have the same loop structure."
    ),
    "surface_area": (
        "Shapes with high surface area have more visual detail to process. "
        "Focus on the overall form before examining surface details."
    ),
    "perspective_mode": (
        "Perspective tasks require imagining what the shape looks like from a different "
        "position. Identify distinctive features (branches, corners), then imagine walking "
        "around to the new viewpoint. Which features would be visible or hidden?"
    ),
    "interaction_strategy": (
        "Try comparing shapes from the current viewpoint before using orbit controls. "
        "Mental rotation builds stronger spatial reasoning than physically rotating. "
        "Compare the overall silhouette first, then verify specific features if needed."
    ),
    "distractor_similarity": (
        "Pay attention to how the parts of the shape connect to each other. "
        "Two shapes can have the same parts but arranged differently — "
        "encode WHERE each arm extends from the branch point, not just what parts exist."
    ),
}


def get_fallback_coaching(weak_areas: List[Dict]) -> str:
    """Generate rule-based coaching from weak areas when LLM is unavailable."""
    if not weak_areas:
        return (
            "You're doing well! Keep practicing to build stronger spatial "
            "reasoning skills. Try increasing the difficulty or switching to "
            "delayed matching mode for an extra challenge."
        )

    messages = ["Based on your performance, here are some tips:\n"]
    for area in weak_areas[:3]:
        feature = area.get("feature", "")
        rate = area.get("success_rate", 0)
        tip = FALLBACK_MESSAGES.get(
            feature,
            f"You're finding shapes with this property challenging ({rate}% accuracy). "
            "Try slowing down and examining shapes more carefully before selecting.",
        )
        messages.append(f"- **{feature}** ({rate}% accuracy): {tip}")

    return "\n".join(messages)
