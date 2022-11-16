import numpy as np
from typing import List
from revolve2.actor_controller import ActorController


class LinearController(ActorController):
    def __init__(self, policy):
        self.policy = policy
        self.state = np.zeros(policy.shape[1])

    def step(self, qpos, qvel, dt: float) -> None:
        input = np.concatenate([qpos, qvel]).ravel()
        self.state = np.matmul(input, self.policy)

    def get_dof_targets(self) -> List[float]:
        return self.state.tolist()

    @staticmethod
    def get_input_size(dof_size: int) -> int:
        # TODO: calculate number of elements from qpos & qval
        return 29
        # return dof_size * 4 + 1

    def serialize(self):
        pass

    @classmethod
    def deserialize(cls, data):
        pass
