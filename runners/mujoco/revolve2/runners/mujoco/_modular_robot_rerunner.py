"""Rerun(watch) a modular robot in Mujoco."""

from typing import Callable, Tuple

import numpy as np
from pyrr import Quaternion, Vector3
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics.actor import Actor
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from revolve2.runners.mujoco import LocalRunner


class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    _controller: ActorController

    async def rerun(
        self,
        robot: ModularRobot,
        control_frequency: float,
        simulation_time=1000000,
        get_pose: Callable[[Actor], Tuple[Vector3, Quaternion]] = None,
    ):
        """
        Rerun a single robot.

        :param robot: The robot the simulate.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        """
        batch = Batch(
            simulation_time=simulation_time,
            sampling_frequency=0.0001,
            control_frequency=control_frequency,
            control=self._control,
        )

        env, self._controller = ModularRobotRerunner.robot_to_env(robot, get_pose)
        batch.environments.append(env)

        runner = LocalRunner(headless=False)
        await runner.run_batch(batch)

    def _control(
        self, environment_index: int, dt: float, control: ActorControl
    ) -> None:
        self._controller.step(dt)
        control.set_dof_targets(0, self._controller.get_dof_targets())

    @staticmethod
    def robot_to_env(
        robot: ModularRobot,
        get_pose: Callable[[Actor], Tuple[Vector3, Quaternion]] = None,
    ) -> Environment:
        """
        Construct an Environment object and contoller for a single robot.

        params:
            robot: ModularRobot to create Environment for
            get_pose: optional function for computing the initial pose of the robot
        """
        actor, controller = robot.make_actor_and_controller()
        env = Environment()
        pos, rot = Vector3([0.0, 0.0, 0.1]), Quaternion()
        if get_pose is not None:
            pos, rot = get_pose(actor)
        env.actors.append(
            PosedActor(
                actor,
                pos,
                rot,
                [0.0 for _ in controller.get_dof_targets()],
            )
        )
        return env, controller


if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )
