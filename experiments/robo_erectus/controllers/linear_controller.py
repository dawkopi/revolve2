import numpy as np
from typing import List
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorState


class LinearController(ActorController):
    def __init__(self, policy):
        self.policy = policy
        self.state = np.zeros(policy.shape[1])

    def step(self, state: ActorState, dt: float) -> None:
        inputs = np.concatenate(
            [state.position, state.orientation, state.hinge_angles, state.hinge_vels]
        ).ravel()
        self.state = np.matmul(inputs, self.policy)

    def get_dof_targets(self) -> List[float]:
        return self.state.tolist()

    @staticmethod
    def get_input_size(dof_size: int) -> int:
        # note: len(position) == 3, len(orientation) == 4
        #   and each hinge has 2 value (pos and vel)
        return 3 + 4 + dof_size * 2

    def serialize(self):
        pass

    @classmethod
    def deserialize(cls, data):
        pass
