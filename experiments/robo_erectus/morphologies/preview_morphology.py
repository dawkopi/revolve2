"""Visualize and run a modular robot using Mujoco."""
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "experiments/robo_erectus"))

from genotypes.linear_controller_genotype import LinearControllerGenotype
from revolve2.runners.mujoco import ModularRobotRerunner

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

from utilities import (
    actor_get_default_pose,
    actor_get_standing_pose,
)


async def main() -> None:
    """Run the simulation."""
    body_name = (
        "erectus_10-10-10"  # register in morphology.py/MORPHOLOGIES before using it
    )
    genotype = LinearControllerGenotype.random(body_name)

    rerunner = ModularRobotRerunner()

    pose_getter = actor_get_standing_pose  # or actor_get_default_pose
    actor, controller = genotype.develop()
    await rerunner.rerun(
        actor,
        controller,
        60,
        simulation_time=60,
        get_pose=pose_getter,
        video_path="",
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
