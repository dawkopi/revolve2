"""Provide various fitness functions for misc. goals."""
import measures
import numpy as np
from revolve2.core.physics.running._results import EnvironmentResults

import measures


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


def with_control_cost(environment_results: EnvironmentResults) -> float:
    base_fitness = measures.displacement_measure(environment_results)
    control_cost = 1e-1 * measures.control_cost(environment_results)
    print(base_fitness, control_cost, base_fitness - control_cost)
    return base_fitness - control_cost


def with_control_height_cost(environment_results: EnvironmentResults) -> float:
    base_fitness = measures.displacement_measure(environment_results)
    control_cost = 1e0 * measures.control_cost(environment_results)
    fintess = (
        base_fitness
        / (1 + measures.max_height_relative_to_avg_height_measure(environment_results))
        - control_cost
    )
    print(base_fitness, control_cost, fintess)
    return fintess


fitness_functions = {
    "displacement_height_groundcontact": displacement_height_groundcontact,
    "displacement_height": displacement_height,
    "displacement_only": displacement_only,
    "with_control_cost": with_control_cost,
    "with_control_height_cost": with_control_height_cost,
}
