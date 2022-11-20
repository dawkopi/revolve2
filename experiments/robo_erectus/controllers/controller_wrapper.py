from revolve2.core.physics.running import ActorControl


class ControllerWrapper:
    controller = None

    def __init__(self, controller):
        self.controller = controller

    def _control(
        self,
        environment_index: int,
        hinge_angles,
        hinge_vels,
        dt: float,
        control: ActorControl,
    ) -> None:
        self.controller.step(hinge_angles, hinge_vels, dt)
        control.set_dof_targets(0, self.controller.get_dof_targets())
