"""
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
"""
import os
import sys
import time
from copy import deepcopy

sys.path.append(
    os.path.join(os.getcwd(), "experiments", "robo_erectus", "optimizers", "ars")
)

import numpy as np
import logz
import utils
import ars_optimizers
from policies import *
from shared_noise import *


class Worker(object):
    """
    Object class for parallel rollout generation.
    """

    def __init__(
        self,
        env_seed,
        reward_func,
        policy_params=None,
        deltas=None,
        rollout_length=1000,
        delta_std=0.02,
    ):
        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params["type"] == "linear":
            self.policy = LinearPolicy(policy_params)
        else:
            raise NotImplementedError

        self.delta_std = delta_std
        self.rollout_length = rollout_length
        self.reward_func = reward_func

    def get_weights_plus_stats(self):
        """
        Get current policy weights and current statistics of past states.
        """
        assert self.policy_params["type"] == "linear"
        return self.policy.get_weights_plus_stats()

    async def rollout(self, inputs, shift=0.0, rollout_length=None):
        """
        Performs one rollout of maximum length rollout_length.
        At each time-step it substracts shift from the reward.
        """

        if rollout_length is None:
            rollout_length = self.rollout_length

        genotypes, database, process_id_gen_gen, process_id_gen = inputs
        fitness, environment_results = await self.reward_func(
            genotypes, database, process_id_gen_gen, process_id_gen
        )
        total_rewards = fitness
        steps_list = [
            len(env_results.environment_states) for env_results in environment_results
        ]
        return total_rewards, steps_list

    async def do_rollouts(
        self, inputs, w_policy, num_rollouts=1, shift=1, evaluate=False
    ):
        """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx = [], []
        steps = 0

        inputs = list(inputs)
        parent = inputs[0]
        inputs[0] = [deepcopy(parent) for i in range(num_rollouts * 2)]
        if evaluate:
            for i in range(num_rollouts):
                inputs[0] = inputs[0][:num_rollouts]
                self.policy.update_weights(w_policy)
                inputs[0][i].genotype = deepcopy(self.policy.get_weights())
                deltas_idx.append(-1)

                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

            # for evaluation we do not shift the rewards (shift = 0) and we use the
            # default rollout length (1000 for the MuJoCo locomotion tasks)
            rewards, steps_list = self.rollout(inputs, shift=0.0, rollout_length=50)
            rollout_rewards = rewards

        else:
            for i in range(num_rollouts):
                idx, delta = self.deltas.get_delta(w_policy.size)

                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                inputs[0][i].genotype = deepcopy(self.policy.get_weights())

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                inputs[0][i + num_rollouts].genotype = deepcopy(
                    self.policy.get_weights()
                )

            rewards, steps_list = await self.rollout(inputs, shift=shift)

            for k in range(num_rollouts):
                pos_reward, neg_reward = rewards[k], rewards[k + num_rollouts]
                pos_steps, neg_steps = steps_list[k], steps_list[k + num_rollouts]
                steps += pos_steps + neg_steps
                rollout_rewards.append([pos_reward, neg_reward])

        return {
            "deltas_idx": deltas_idx,
            "rollout_rewards": rollout_rewards,
            "steps": steps,
        }

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return


class ARSLearner(object):
    """
    Object class implementing the ARS algorithm.
    """

    def __init__(
        self,
        x0,
        reward_func=None,
        policy_params=None,
        num_workers=32,
        num_deltas=320,
        deltas_used=320,
        delta_std=0.02,
        logdir=None,
        rollout_length=1000,
        step_size=0.01,
        shift="constant zero",
        params=None,
        seed=123,
    ):

        logz.configure_output_dir(logdir)
        logz.save_params(params)

        self.timesteps = 0
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params

        # create shared table for storing noise
        print("Creating deltas table.")
        noise = create_shared_noise()
        self.deltas = SharedNoiseTable(noise, seed=seed + 3)
        print("Created deltas table.")

        # initialize workers with different random seeds
        print("Initializing workers.")
        self.num_workers = num_workers
        self.workers = [
            Worker(
                seed + 7 * i,
                reward_func=reward_func,
                policy_params=policy_params,
                deltas=noise,
                rollout_length=rollout_length,
                delta_std=delta_std,
            )
            for i in range(num_workers)
        ]

        # initialize policy
        if policy_params["type"] == "linear":
            self.policy = LinearPolicy(policy_params)
            if policy_params["random_initiate"]:
                self.w_policy = x0
                self.policy.weights = x0
            else:
                self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError

        # initialize optimization algorithm
        self.optimizer = ars_optimizers.SGD(self.w_policy, self.step_size)
        print("Initialization of ARS complete.")

    async def aggregate_rollouts(self, inputs, num_rollouts=None, evaluate=False):
        """
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)

        # parallel generation of rollouts
        rollout_ids_one = [
            await worker.do_rollouts(
                inputs,
                self.w_policy,
                num_rollouts=num_rollouts,
                shift=self.shift,
                evaluate=evaluate,
            )
            for worker in self.workers
        ]

        rollout_ids_two = [
            await worker.do_rollouts(
                inputs,
                self.w_policy,
                num_rollouts=1,
                shift=self.shift,
                evaluate=evaluate,
            )
            for worker in self.workers[: (num_deltas % self.num_workers)]
        ]

        # gather results
        results_one = rollout_ids_one
        results_two = rollout_ids_two

        rollout_rewards, deltas_idx = [], []

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result["deltas_idx"]
            rollout_rewards += result["rollout_rewards"]

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result["deltas_idx"]
            rollout_rewards += result["rollout_rewards"]

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)

        print("Maximum reward of collected rollouts:", rollout_rewards.max())
        t2 = time.time()

        print("Time to generate rollouts:", t2 - t1)

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis=1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas

        idx = np.arange(max_rewards.size)[
            max_rewards
            >= np.percentile(
                max_rewards, 100 * (1 - (self.deltas_used / self.num_deltas))
            )
        ]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(
            rollout_rewards[:, 0] - rollout_rewards[:, 1],
            (self.deltas.get(idx, self.w_policy.size) for idx in deltas_idx),
            batch_size=500,
        )
        g_hat /= deltas_idx.size
        t2 = time.time()
        print("time to aggregate rollouts", t2 - t1)
        return g_hat

    async def train_step(self, inputs):
        """
        Perform one update step of the policy weights.
        """

        g_hat = await self.aggregate_rollouts(inputs)
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(
            self.w_policy.shape
        )  # TODO update weights

        # get statistics from all workers
        for j in range(self.num_workers):
            self.policy.observation_filter.update(self.workers[j].get_filter())
        self.policy.observation_filter.stats_increment()

        # make sure master filter buffer is clear
        self.policy.observation_filter.clear_buffer()
        # sync all workers
        filter_id = self.policy.observation_filter
        [worker.sync_filter(filter_id) for worker in self.workers]

        [worker.stats_increment() for worker in self.workers]
        return
