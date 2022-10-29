from revolve2.core.physics.running._results import EnvironmentResults
import measures


def displacement_height_groundcontact(environment_results: EnvironmentResults) -> float:
    return (
        measures.ground_contact_measure(environment_results)
        * measures.displacement_measure(environment_results)
        / measures.max_height_relative_to_avg_height_measure(environment_results)
    )


def displacement_only(environment_results: EnvironmentResults) -> float:
    return measures.displacement_measure(environment_results)


def axis_displacement_only(environment_results: EnvironmentResults) -> float:
    return measures.x_displacement_measure(environment_results) - abs(
        measures.y_displacement_measure(environment_results)
    )


def directed_diagonal_displacement(environment_results: EnvironmentResults) -> float:
    # displacement = measures.displacement_measure_cont(environment_results)
    # fitness = []
    # for d in displacement:
    #     x = d[0]
    #     y = d[1]
    #     if x > 0 and y > 0:
    #         ratio = x / y if x < y else y / x
    #         fitness.append((x + y) * ratio)
    # return float(sum(fitness))
    return measures.x_displacement_measure(
        environment_results
    ) + measures.y_displacement_measure(environment_results)


def displacement_height(environment_results: EnvironmentResults) -> float:
    return measures.displacement_measure(
        environment_results
    ) / measures.max_height_relative_to_avg_height_measure(environment_results)


def axis_displacement_height(environment_results: EnvironmentResults) -> float:
    return (
        measures.x_displacement_measure(environment_results)
        - abs(measures.y_displacement_measure(environment_results))
    ) / measures.max_height_relative_to_avg_height_measure(environment_results)


fitness_functions = {
    "displacement_height_groundcontact": displacement_height_groundcontact,
    "displacement_only": displacement_only,
    "axis_displacement_only": axis_displacement_only,
    "directed_diagonal_displacement": directed_diagonal_displacement,
    "displacement_height": displacement_height,
    "axis_displacement_height": axis_displacement_height,
}
