import numpy as np
from typing import List
from revolve2.actor_controller import ActorController


class LinearController(ActorController):
    def __init__(self, policy):
        self.policy = policy
        self.state = np.zeros(policy.shape[1])

    def step(self, hinge_angles, hinge_vels, dt: float) -> None:
        inputs = np.concatenate([hinge_angles, hinge_vels]).ravel()
        self.state = np.matmul(inputs, self.policy)

    def get_dof_targets(self) -> List[float]:
        return self.state.tolist()

    @staticmethod
    def get_input_size(dof_size: int) -> int:
        return dof_size * 2

    def serialize(self):
        pass

    @classmethod
    def deserialize(cls, data):
        pass
