from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from pyrr import Quaternion, Vector3


@dataclass
class ActorState:
    """State of an actor."""

    position: Vector3
    orientation: Quaternion

    # IDs of geometries (of this actor) in contact with ground
    groundcontacts: Set[int] | None = None
    # count of total geometries in Actor's morphology
    numgeoms: int | None = None

    # angles of each joint
    qpos: List[float] | None = None
    # velocities of each joint
    qvel: List[float] | None = None

    # angles of each joint
    qpos: List[float] = None
    # velocities of each joint
    qvel: List[float] = None


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


@dataclass
class BatchResults:
    """Result of running a batch."""

    environment_results: List[EnvironmentResults]
