from revolve2.core.physics.running._results import EnvironmentResults
import measures


def displacement_height_groundcontact(
    environment_results: EnvironmentResults, bias: float = 0.0
) -> float:
    return (
        (measures.ground_contact_measure(environment_results) + bias)
        * measures.displacement_measure(environment_results)
        / measures.max_height_relative_to_avg_height_measure(environment_results)
    )


def displacement_only(environment_results: EnvironmentResults) -> float:
    return measures.displacement_measure(environment_results)


fitness_functions = {
    "displacement_height_groundcontact": displacement_height_groundcontact,
    "displacement_height_groundcontact_biased": lambda r: displacement_height_groundcontact(
        r, 1.0
    ),
    "displacement_only": displacement_only,
}
