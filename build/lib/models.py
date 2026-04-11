"""
DataBridgeEnv — Pydantic models for Action, Observation, and Reward.

The agent sees a broken payload and a target schema, then submits a
transformer (Python dict) that should map the broken payload to the schema.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TransformAction(BaseModel):
    """
    The agent's proposed fix.

    The agent submits a `transformed_payload` — a Python dict it believes
    matches the target schema.  Optionally it can explain its reasoning.
    """
    transformed_payload: Dict[str, Any] = Field(
        ...,
        description="The agent's corrected payload that should match target_schema.",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Optional chain-of-thought explaining what was wrong and how it was fixed.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ContractObservation(BaseModel):
    """
    Everything the agent sees at the start of (and during) an episode.
    """
    task_id: str = Field(..., description="Which task: 'easy' | 'medium' | 'hard'")
    episode_id: str = Field(..., description="Unique episode identifier (UUID).")

    # The upstream broken payload the agent must fix
    broken_payload: Dict[str, Any] = Field(
        ...,
        description="The malformed payload arriving from the upstream service.",
    )

    # For easy & medium: target schema is explicit
    # For hard: this is None — agent must infer from examples
    target_schema: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Explicit target schema (field -> expected type string). "
            "None for the hard task — agent must infer from examples."
        ),
    )

    # For hard task: a list of (input, correct_output) pairs to infer the schema
    schema_examples: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Example (broken_input, correct_output) pairs for schema inference. "
            "Only provided for the hard task."
        ),
    )

    # Mismatch hints — tells the agent *which categories* of bugs exist
    # (always provided so the agent doesn't have to do blind search)
    mismatch_hints: List[str] = Field(
        default_factory=list,
        description=(
            "High-level hints about what kinds of mismatches are present. "
            "E.g. ['field_name', 'type_coercion', 'array_scalar']"
        ),
    )

    # Step metadata
    step_number: int = Field(0, description="Current step within the episode (0-indexed).")
    max_steps: int = Field(3, description="Maximum steps allowed per episode.")
    done: bool = Field(False, description="Whether the episode has ended.")

    # Reward from the previous step (None on the first observation)
    last_reward: Optional[float] = Field(
        None,
        description="Reward from the previous action (None before first step).",
    )

    # Feedback from the previous step (None on the first observation)
    last_feedback: Optional[str] = Field(
        None,
        description="Human-readable feedback from the grader on the previous attempt.",
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class ContractReward(BaseModel):
    """
    Granular per-field reward breakdown.

    score is in [0.0, 1.0] and is the mean of per-field scores.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Overall score 0.0–1.0.")

    field_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-field score: 1.0 = fully correct, 0.0 = wrong/missing.",
    )

    # Breakdown by mismatch category (how many of each type were fixed)
    fixed_field_names: int = Field(0, description="Number of field-name mismatches resolved.")
    fixed_type_coercions: int = Field(0, description="Number of type-coercion mismatches resolved.")
    fixed_shapes: int = Field(0, description="Number of shape/nesting mismatches resolved.")
    fixed_array_scalars: int = Field(0, description="Number of array/scalar mismatches resolved.")
    fixed_missing_fields: int = Field(0, description="Number of missing/extra fields resolved.")

    total_fields: int = Field(0, description="Total fields in the target schema.")
    correct_fields: int = Field(0, description="Fields that matched the target exactly.")

    done: bool = Field(False, description="True when episode is over (success or max steps).")
    success: bool = Field(False, description="True when score == 1.0.")
    feedback: str = Field("", description="Human-readable explanation of the score.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class BridgeState(BaseModel):
    """
    Episode-level metadata returned by GET /state.
    """
    episode_id: str
    task_id: str
    step_number: int
    max_steps: int
    done: bool
    current_score: float = 0.0
    mismatch_types_present: List[str] = Field(default_factory=list)
    total_fields: int = 0
