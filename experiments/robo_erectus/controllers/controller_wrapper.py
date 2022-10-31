from revolve2.core.physics.running import ActorControl


class ControllerWrapper:
    controller = None

    def __init__(self, controller):
        self.controller = controller

    def _control(
        self, environment_index: int, dt: float, control: ActorControl
    ) -> None:
        self.controller.step(dt)
        control.set_dof_targets(0, self.controller.get_dof_targets())
