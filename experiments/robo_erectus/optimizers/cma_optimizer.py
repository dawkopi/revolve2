import os
import sys
import cma
import logging

sys.path.append(os.getcwd())

import wandb
from measures import *
from optimizer import Optimizer


class CmaEsOptimizer(Optimizer):
    def init_cma(self, offspring_size):
        logging.info(f"Initalizing CMA-ES...")
        init_x = None
        for individual in self._latest_population:
            genotype = individual.genotype.genotype
            body_name = individual.genotype.body_name
            init_x = genotype
        self._es = cma.CMAEvolutionStrategy(init_x, 0.2, {"popsize": offspring_size})
        return body_name

    async def evovle_step(
        self,
        body_name,
        database,
        process_id_gen,
        genotype_type,
        Individual,
        safe_evaluate_generation,
        gen_next_individual_id,
    ):
        # let user create offspring
        solutions = self._es.ask()

        offspring = [genotype_type(solution, body_name) for solution in solutions]

        # let user evaluate offspring
        new_fitnesses, new_results = await safe_evaluate_generation(
            offspring, database, process_id_gen.gen(), process_id_gen
        )

        # cma-es minimize fitness function, need to convert max-fitness to min-fitness
        fitnesses_for_selection = [1 / fitness for fitness in new_fitnesses]
        self._es.tell(solutions, fitnesses_for_selection)
        # self._es.disp()

        survived_new_individuals = [
            Individual(
                -1,  # placeholder until later
                genotype_type(solution, body_name),
                [i for i in range(len(solutions))],
            )
            for index, solution in enumerate(solutions)
        ]

        survived_new_fitnesses = new_fitnesses
        survived_new_results = new_results

        # set ids for new individuals
        for individual in survived_new_individuals:
            individual.id = gen_next_individual_id()

        # combine old and new and store as the new generation
        self._latest_population = survived_new_individuals
        self._latest_fitnesses = survived_new_fitnesses
        self._latest_results = survived_new_results

        return survived_new_individuals, survived_new_fitnesses

    def _log_results(self) -> None:
        displacement = [displacement_measure(r) for r in self._latest_results]
        steps = [len(r.environment_states) for r in self._latest_results]

        wandb.log(
            {
                "steps_max": max(steps),
                "steps_avg": sum(steps) / len(steps),
                "steps_min": min(steps),
                "steps": wandb.Histogram(steps),
                "displacement_max": max(displacement),
                "displacement_avg": sum(displacement) / len(displacement),
                "displacement_min": min(displacement),
                "fitness_max": max(self._latest_fitnesses),
                "fitness_avg": sum(self._latest_fitnesses)
                / len(self._latest_fitnesses),
                "fitness_min": min(self._latest_fitnesses),
                "displacement": wandb.Histogram(displacement),
                "max_height_relative_to_avg_height": wandb.Histogram(
                    [
                        max_height_relative_to_avg_height_measure(r)
                        for r in self._latest_results
                    ]
                ),
                "ground_contact_measure": wandb.Histogram(
                    [ground_contact_measure(r) for r in self._latest_results]
                ),
            }
        )
