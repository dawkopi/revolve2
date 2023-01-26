import os
import sys
import time
import logging
from copy import deepcopy

sys.path.append(os.getcwd())

import wandb
from measures import *
from optimizer import Optimizer
from .ars.ars import ARSLearner


class ArsOptimizer(Optimizer):
    def init_optimizer(self, param):
        Genotype = param[1]
        evaluate_func = param[2]

        params = {  # was working well for 64,64 + 4 resamples and 8,4 + 8 resampless
            "n_directions": 64,  # evolve direction (like number of children / 2)
            "deltas_used": 64,  # like how many survivors
            "step_size": 0.02,  # evolve step size
            "delta_std": 0.03,
            "n_workers": 1,  # number of cpus
            "rollout_length": 1000,  # simulation steps
            "shift": 0,
            "seed": 237,
            "policy_type": "linear",
            "dir_path": "data",
            "filter": "NoFilter",
            # "filter": "MeanStdFilter", # Dawid worries about how well this is implemented for us / if its necessary
        }
        if self.n_directions != None:
            params["n_directions"] = self.n_directions
            params["deltas_used"] = self.n_directions
        if self.step_size != None:
            params["step_size"] = self.step_size
        logging.info(f"Initalizing ARS... (n_directions={params['n_directions']})")

        dir_path = params["dir_path"]

        if not (os.path.exists(dir_path)):
            os.makedirs(dir_path)
        logdir = dir_path
        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        init_x = None
        for individual in self._latest_population:
            genotype = individual.genotype.genotype
            body_name = individual.genotype.body_name
            init_x = genotype

        _, dof_ids = Genotype.develop_body(body_name)
        ac_dim = len(dof_ids)
        ob_dim = init_x.shape[0] // ac_dim

        # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
        policy_params = {
            "type": "linear",
            "ob_filter": params["filter"],
            "ob_dim": ac_dim,
            "ac_dim": ob_dim,
            "random_initiate": False,
        }

        self._ars = ARSLearner(
            x0=init_x,
            reward_func=evaluate_func,
            policy_params=policy_params,
            num_workers=params["n_workers"],
            num_deltas=params["n_directions"],
            deltas_used=params["deltas_used"],
            step_size=params["step_size"],
            delta_std=params["delta_std"],
            logdir=logdir,
            rollout_length=params["rollout_length"],
            shift=params["shift"],
            params=params,
            seed=params["seed"],
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
        parent = genotype_type(deepcopy(self._ars.w_policy), body_name)
        inputs = (parent, database, process_id_gen.gen(), process_id_gen)

        # let user create offspring
        await self._ars.train_step(inputs)
        solutions = [self._ars.w_policy]

        offspring = [genotype_type(solution, body_name) for solution in solutions]

        survived_new_individuals = [
            Individual(
                -1,  # placeholder until later
                genotype_type(solution, body_name),
                [i for i in range(len(solutions))],
            )
            for index, solution in enumerate(solutions)
        ]

        # let user evaluate offspring
        new_fitnesses, new_results = await safe_evaluate_generation(
            offspring, database, process_id_gen.gen(), process_id_gen
        )

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
                "sim_step": self._unique_sim_steps,
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
