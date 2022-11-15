"""Provide various fitness functions for misc. goals."""
import measures
import numpy as np
from revolve2.core.physics.running._results import EnvironmentResults

import measures

control_cost_weight = 0.5


def displacement_height_groundcontact(environment_results: EnvironmentResults) -> float:
    """TODO."""
    return (
        measures.ground_contact_measure(environment_results)
        * measures.displacement_measure(environment_results)
        / measures.max_height_relative_to_avg_height_measure(environment_results)
    )


def displacement_height(environment_results: EnvironmentResults) -> float:
    """TODO."""
    return measures.displacement_measure(
        environment_results
    ) / measures.max_height_relative_to_avg_height_measure(environment_results)


def displacement_only(environment_results: EnvironmentResults) -> float:
    """TODO."""
    return measures.displacement_measure(environment_results)


def directed_displacement_only(environment_results: EnvironmentResults) -> float:
    return measures.directed_displacement_measure(environment_results)


def displacement_with_height_reward_and_control_cost(
    environment_results: EnvironmentResults,
) -> float:
    base_fitness = measures.displacement_measure(environment_results)
    height_reward = measures.average_height_measure(environment_results) ** 2
    control_cost = control_cost_weight * measures.control_cost(environment_results)
    return base_fitness + height_reward - control_cost


def displacement_with_height_reward(environment_results: EnvironmentResults) -> float:
    return (
        measures.displacement_measure(environment_results)
        + measures.average_height_measure(environment_results) ** 2
    )


def with_control_cost(environment_results: EnvironmentResults) -> float:
    base_fitness = measures.displacement_measure(environment_results)
    control_cost = control_cost_weight * measures.control_cost(environment_results)
    print(base_fitness, control_cost)
    return base_fitness - control_cost


def directed_with_control_cost(environment_results: EnvironmentResults) -> float:
    base_fitness = measures.directed_displacement_measure(environment_results)
    control_cost = control_cost_weight * measures.control_cost(environment_results)
    return base_fitness - control_cost


def with_control_height_cost(environment_results: EnvironmentResults) -> float:
    base_fitness = measures.displacement_measure(environment_results)
    control_cost = control_cost_weight * measures.control_cost(environment_results)
    fintess = (
        base_fitness
        / (1 + measures.max_height_relative_to_avg_height_measure(environment_results))
        - control_cost
    )
    return fintess


fitness_functions = {
    "displacement_height_groundcontact": displacement_height_groundcontact,
    "displacement_height": displacement_height,
    "displacement_only": displacement_only,
    "with_control_cost": with_control_cost,
    "with_control_height_cost": with_control_height_cost,
    "directed_displacement_only": directed_displacement_only,
    "directed_with_control_cost": directed_with_control_cost,
    "displacement_with_height_reward": displacement_with_height_reward,
    "displacement_with_height_reward_and_control_cost": displacement_with_height_reward_and_control_cost,
}
