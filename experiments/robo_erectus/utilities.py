"""Provide misc helper functions."""
from genericpath import isfile
import os
import numpy as np
from pyrr import Quaternion, Vector3
from revolve2.core.physics.actor import Actor
from revolve2.core.physics.running._results import ActorState
from typing import Tuple, Optional
from typing_extensions import LiteralString

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

DATABASE_PATH = os.path.join(SCRIPT_DIR, "database")
ANALYSIS_DIR_NAME = "analysis"
LASTEST_RUN_FILENAME = "latest"


def ensure_dirs(*dir_paths):
    """Ensure a list of directories all exist (creating them as needed)."""
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


def find_dir(parent_path: str, target_name: str) -> str:
    """Find a directory by name within a given path."""
    target_dir = os.path.join(parent_path, target_name)
    if not os.path.isdir(target_dir):
        for dir_name in sorted(os.listdir(parent_path)):
            if dir_name.startswith(target_name):
                return dir_name
        return None
    else:
        return target_name


def get_latest_run() -> str:
    """Get full name of latest experiment run."""
    target = os.path.join(DATABASE_PATH, LASTEST_RUN_FILENAME)
    if os.path.isfile(target):
        with open(target, "r") as file:
            full_run_name = file.read().rstrip()
            return full_run_name
    else:
        return None


def set_latest_run(full_run_name: str):
    """Cache the name of the latest run."""
    target = os.path.join(DATABASE_PATH, LASTEST_RUN_FILENAME)
    with open(target, "w") as file:
        file.write(full_run_name)


def get_random_rotation(scale=0.02) -> Quaternion:
    rot = Quaternion.from_x_rotation(np.random.uniform(-1.0, 1.0) * np.pi * scale)
    rot = rot * Quaternion.from_y_rotation(np.random.uniform(-1.0, 1.0) * np.pi * scale)
    rot = rot * Quaternion.from_z_rotation(np.random.uniform(-1.0, 1.0) * np.pi * scale)
    return rot


# TODO: add param for tweaking the initial pose (making it stochastic)
def actor_get_standing_pose(actor: Actor) -> Tuple[Vector3, Quaternion]:
    """
    Given an actor, return a pose (such that it starts out "standing" upright).

    Returns tuple (pos, rot).
    """
    bounding_box = actor.calc_aabb()
    pos = Vector3(
        [
            0.0,
            0.0,
            # due to rotating about the y axis, the box's x size becomes the new effective "z" height of the box
            bounding_box.size.x / 2.0 - bounding_box.offset.x + 0.1,
        ]
    )

    rot = Quaternion.from_y_rotation(np.pi / 2)
    rot = rot * get_random_rotation()

    return (pos, rot)


def actor_get_default_pose(actor: Actor) -> Tuple[Vector3, Quaternion]:
    """Original method of computing initial pose for an Actor (so it starts "flat" on the ground)."""
    bounding_box = actor.calc_aabb()
    pos = Vector3(
        [
            0.0,
            0.0,
            bounding_box.size.z / 2.0 - bounding_box.offset.z + 0.1,
        ]
    )

    rot = Quaternion()
    rot = rot * get_random_rotation()

    return (pos, rot)


def is_healthy_state(
    state: ActorState, min_z: float, max_z: Optional[float] = None
) -> bool:
    """
    Indicates whether actor is in a healthy state (e.g. hasn't fallen over).
    Pass this function a min_z value and (optionally) a max_z value.

    references:
        https://www.gymlibrary.dev/environments/mujoco/hopper/#episode-end
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/hopper_v4.py

        https://gymnasium.farama.org/environments/mujoco/humanoid/#episode-end
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/humanoid_v4.py
    """

    if min_z is None and max_z is None:
        return True

    healthy = state.position.z >= min_z
    if max_z is not None:
        healthy = healthy and state.position.z <= max_z

    return healthy
