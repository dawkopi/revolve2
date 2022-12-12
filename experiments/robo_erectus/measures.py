"""Implement some functions for measuring fitness aspects."""
import logging
import math
import numpy as np
from typing import Union

from numpy import average
from revolve2.core.physics.running._results import EnvironmentResults


def control_cost(environment_results: EnvironmentResults) -> float:
    action_diffs = sum(
        [state.action_diffs for state in environment_results.environment_states], []
    )
    control_costs = [np.sum(np.square(diff)) for diff in action_diffs]
    return float(np.sum(control_costs))


# def control_cost(environment_results: EnvironmentResults) -> float:
#     actions = sum(
#         [state.actions for state in environment_results.environment_states], []
#     )
#     control_costs = [np.sum(np.square(action)) for action in actions]
#     return float(np.sum(control_costs))


def directed_displacement_measure(environment_results: EnvironmentResults) -> float:
    begin_state = environment_results.environment_states[0].actor_states[0]
    end_state = environment_results.environment_states[-1].actor_states[0]
    distance = (begin_state.position[0] - end_state.position[0]) ** 2
    distance = distance - abs(begin_state.position[1] - end_state.position[1])
    return float(distance)


def displacement_measure(environment_results: EnvironmentResults) -> float:
    """Measure how far robot moved from start to end of simulation."""
    begin_state = environment_results.environment_states[0].actor_states[0]
    end_state = environment_results.environment_states[-1].actor_states[0]
    distance = math.sqrt(
        (begin_state.position[0] - end_state.position[0]) ** 2
        + ((begin_state.position[1] - end_state.position[1]) ** 2)
    )
    return float(distance)


def average_height_measure(environment_results: EnvironmentResults) -> float:
    heights = [
        s.actor_states[0].position[2] for s in environment_results.environment_states
    ]
    return float(sum(heights) / len(heights))


def max_height_relative_to_avg_height_measure(
    environment_results: EnvironmentResults,
) -> float:
    """Attempt to discourage jumping behavior."""
    heights = [
        environment_results.environment_states[i + 1].actor_states[0].position[2]
        for i in range(len(environment_results.environment_states) - 1)
    ]
    return float(max(heights) / (sum(heights) / len(heights)))


def ground_contact_measure(
    environment_results: EnvironmentResults,
) -> Union[float, None]:
    """
    Return a score in [0,1] indicating how well contact with the ground was minimized (1.0 being ideal).

    The percent of time each geometry within the Actor was in contact with the ground is tabulated.
    The top two geometries with the most ground contact are considered the "feet" and aren't penalized.
    """
    geom_data = (
        {}
    )  # map geom IDs to a count of samples in which they were touching ground
    actor_states = [
        env_state.actor_states[0]
        for env_state in environment_results.environment_states
    ]
    for a in actor_states:
        if a.groundcontacts == None or a.numgeoms == None:
            return None  # in case necessary data wasn't tracked
        for geomid in a.groundcontacts:
            if geomid not in geom_data:
                geom_data[geomid] = 0
            geom_data[geomid] += 1

    numgeoms = actor_states[0].numgeoms
    if numgeoms <= 2 or len(geom_data) <= 2:
        return 1.0  # nothing to penalize

    # convert to a float (ratio of time at which geom was in contact with ground):
    for key in geom_data:
        geom_data[key] /= len(actor_states)

    # sort geom IDs by most contact with ground (descending) to least
    ranked_ids = sorted(geom_data.keys(), key=lambda id: -geom_data[id])

    ranked_nonfeet = ranked_ids[2:]

    logging.debug(f"tabulated geom_data (num actor_states = {len(actor_states)}):")
    logging.debug(geom_data)
    # weights determining important of penalizing most active non-foot, vs penalizing average of ALL non-feet
    w1, w2 = (0.6, 0.4)  # should sum to 1.0

    # get average ratio of time each non-foot geom is in contact with ground
    #   (considering also that geoms that never touch ground won't be in geom_data)
    average_non_feet = average(
        [geom_data[id] for id in ranked_nonfeet]
        + [0.0 for _ in range(numgeoms - len(geom_data))]
    )
    # take weighted average such that result will be in [0, 1]

    score = average([geom_data[ranked_nonfeet[0]], average_non_feet], weights=[w1, w2])
    return 1.0 - float(score)
