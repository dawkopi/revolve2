from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Optional

from pyrr import Quaternion, Vector3


@dataclass
class ActorState:
    """State of an actor."""

    position: Vector3
    orientation: Quaternion

    # IDs of geometries (of this actor) in contact with ground
    groundcontacts: Optional[Set[int]] = None
    # count of total geometries in Actor's morphology
    numgeoms: Optional[int] = None

    # angles of each hinge joint
    hinge_angles: List[float] | None = None
    # velocities of each hinge joint
    hinge_vels: List[float] | None = None


@dataclass
class EnvironmentState:
    """State of an environment."""

    time_seconds: float
    actions: List[List[float]]
    action_diffs: List[List[float]]
    actor_states: List[ActorState]


@dataclass
class EnvironmentResults:
    """Result of running an environment."""

    environment_states: List[EnvironmentState]
    steps_completed: int = 0  # total number of simulation steps performed


@dataclass
class BatchResults:
    """Result of running a batch."""

    environment_results: List[EnvironmentResults]
