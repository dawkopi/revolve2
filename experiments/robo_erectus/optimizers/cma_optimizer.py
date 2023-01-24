import os
import sys
import cma
import logging

sys.path.append(os.getcwd())

import wandb
from measures import *
from optimizer import Optimizer


class CmaEsOptimizer(Optimizer):
    def init_optimizer(self, param):
        offspring_size = param[0]
        logging.info(f"Initalizing CMA-ES...")
        init_x = None
        for individual in self._latest_population:
            genotype = individual.genotype.genotype
            body_name = individual.genotype.body_name
            init_x = genotype
        self._es = cma.CMAEvolutionStrategy(
            init_x, self.sigma0, {"popsize": offspring_size}
        )
        return body_name

    async def evolve_step(
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
