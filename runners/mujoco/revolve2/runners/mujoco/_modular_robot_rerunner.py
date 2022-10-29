"""Rerun(watch) a modular robot in Mujoco."""

from pyrr import Quaternion, Vector3
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics.running import (
    ActorControl,
    Batch,
    Environment,
    PosedActor,
    EnvironmentState,
)
from revolve2.runners.mujoco import LocalRunner


class ModularRobotRerunner:
    """Rerunner for a single robot that uses Mujoco."""

    _controller: ActorController

    async def rerun(
        self, robot: ModularRobot, control_frequency: float, simulation_time=1000000
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

        env, self._controller = ModularRobotRerunner.robot_to_env(robot)
        batch.environments.append(env)

        runner = LocalRunner(headless=False)
        await runner.run_batch(batch)

    def _control(
        self,
        environment_index: int,
        dt: float,
        control: ActorControl,
        state: EnvironmentState,
    ) -> None:
        controller = self._controller
        _, dof_ids = controller.body.to_actor()
        state.actor_states[0].dof_targets = list(
            zip(dof_ids, controller.get_dof_targets())
        )
        _controller = controller.brain.make_controller(
            controller.body, controller.dof_ids, state
        )
        controller._weight_matrix = _controller._weight_matrix
        controller.step(dt)
        control.set_dof_targets(0, controller.get_dof_targets())

    @staticmethod
    def robot_to_env(robot: ModularRobot) -> Environment:
        """Constructs an Environment object and contoller for a single robot."""
        from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import (
            develop_v1 as body_develop,
        )

        body = robot.body
        brain = robot.brain

        actor, dof_ids = body.to_actor()
        controller = brain.make_controller(body, dof_ids)
        ##
        controller.body = body
        controller.brain = brain
        controller.dof_ids = dof_ids
        ##

        env = Environment()
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, 0.1]),
                Quaternion(),
                [0.0 for _ in controller.get_dof_targets()],
            )
        )
        return env, controller


if __name__ == "__main__":
    print(
        "This file cannot be ran as a script. Import it and use the contained classes instead."
    )
