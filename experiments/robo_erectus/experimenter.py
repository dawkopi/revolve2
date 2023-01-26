#!/usr/bin/env python3
import argparse
import os
import subprocess
import copy
from typing import List
import yaml
import math
from multiprocessing import cpu_count

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
OPTIMIZE = os.path.join(SCRIPT_DIR, "optimize.py")


def main():
    CPUS = cpu_count()
    parser = argparse.ArgumentParser(
        description=f"Runs a set of experiments by calling {os.path.basename(OPTIMIZE)}"
    )
    # erectus_000
    parser.add_argument("-m", "--morphology", type=str, default="erectus_000")
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="name of experiment (prefixes run names, and names tmux session)",
    )
    parser.add_argument(
        "-cpu",
        "--max_cpus",
        type=int,
        default=int(CPUS / 2),
        help="max number of cpus to use at once (across experiment)",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        default=False,
        help="dry run (just print commands)",
    )
    parser.add_argument(
        "-nw", "--no-wandb", action="store_true", help="don't use wandb"
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="number of times to run each unique command",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--num_generations",
        type=int,
        default=9999,
        help="defaults to arbitrarily high",
    )
    # parser.add_argument("-t", "--simulation_time", type=int, default=10)
    args = parser.parse_args()

    wandb_flags = []
    if not args.no_wandb:
        wandb_flags = ["-w", "--wandb_os_logs"]

    assert os.path.exists(OPTIMIZE)

    task_batch = []  # build list of commands to run
    for opti in ["cma", "ars"]:
        for pop in [10, 100]:
            base_cmd = [
                "time",
                "python3",
                OPTIMIZE,
                "-m",
                args.morphology,
                "-p",
                str(pop),
                "--max_steps",  # millions
                str(500),
                "-g",
                str(args.num_generations),
                "--skip_best",
                *wandb_flags,
                # "-t",
                # str(args.simulation_time),
            ]

            if opti == "cma":
                for sigma0 in [0.2, 0.8]:
                    group_name = f"{args.prefix + '_' if args.prefix else ''}_{opti}_pop{pop}_sigma{sigma0}"
                    for trial in range(args.trials):
                        task_batch.append(
                            [
                                *base_cmd,
                                "-cma",
                                "--sigma0",
                                str(sigma0),
                                "--group_name",
                                group_name,
                                "-n",
                                f"{group_name}_trial{trial}",
                            ]
                        )
            elif opti == "ars":
                for step_size in [0.002, 0.02, 0.2]:
                    group_name = f"{args.prefix + '_' if args.prefix else ''}_{opti}_pop{pop}_step_size{step_size}"
                    for trial in range(args.trials):
                        task_batch.append(
                            [
                                *base_cmd,
                                "-ars",
                                "--step_size",
                                str(step_size),
                                "--group_name",
                                group_name,
                                "-n",
                                f"{group_name}_trial{trial}",
                            ]
                        )

            else:
                raise NotImplementedError

    print(f"collected {len(task_batch)} commands to {'dry' if args.dry else ''} run\n")
    for i, cmd in enumerate(task_batch):
        # run_cmd(cmd, dry_run=args.dry, msg=f"{i+1}/{len(task_batch)}")
        run_cmd(cmd, dry_run=True, msg=f"{i+1}/{len(task_batch)}")

    # TODO: fix this scheduling / calculation of cpus_per_task
    num_tasks = len(task_batch)
    args.max_cpus = min(CPUS, args.max_cpus)
    parallel = 1  # num tasks to run in parallel
    cpus_per_task = 1
    if args.max_cpus > 1:
        # try to give 4 cpus per task
        cpus_per_task = min(1, args.max_cpus)
        parallel = math.floor(args.max_cpus / cpus_per_task)

    # TODO: now go through and add -cpu flag to all tasks?

    print(
        f"\nplan: run {math.ceil(num_tasks / parallel)} batches, each with {cpus_per_task} cpus"
    )

    if not args.dry:
        schedule_jobs(task_batch, args.prefix, parallel)


exit_codes = []


def run_cmd(cmd, msg="", dry_run=False):
    print(
        f"\n{'running' if not dry_run else 'would run'} command{' ' + msg if msg else ''}:"
    )
    print(" ".join(cmd))
    if dry_run:
        return

    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    exit_codes.append(res.returncode)
    if res.returncode != 0:
        print(f"***command failed with code {res.returncode}")
        print(" ".join(cmd))
    return res.returncode


def schedule_jobs(task_batch: List, name: str, parallel: int):
    """
    Schedule jobs to run in a tmux session.
    Requires teamocil and tmux to be installed.  https://github.com/remi/teamocil#installation
    params:
        task_batch: list of commands (where each command is a list strings)
        parallel: max number of tasks to run in parallel (as separate tmux tabs)
        name: name of tmux session to create
    """
    print(f"\nscheduling {len(task_batch)} tasks!")
    tasks = copy.deepcopy(task_batch)
    # create dict defining layour for teamocil
    data = {
        "name": name,
        "windows": [],
    }
    bin_size = math.ceil(len(task_batch) / parallel)
    batch_num = 0
    while len(tasks) > 0:
        # flatten tasks into list of strings
        cur_tasks = tasks[0:bin_size]
        cur_tasks = [" ".join(cmd) for cmd in cur_tasks]
        data["windows"].append(
            {
                "name": f"batch{str(batch_num+1).zfill(2)}",
                "root": SCRIPT_DIR,
                "layout": "tiled",
                # chain commands together to run in series
                "panes": ["date; " + "; date; ".join(cur_tasks)],
                # "panes": [{"commands": tasks[0:bin_size]}],
            }
        )
        tasks = tasks[bin_size:]
        batch_num += 1

    fname = os.path.join(SCRIPT_DIR, f"layout_{name}.yml")
    with open(fname, "w") as f:
        yaml.dump(
            data,
            f,
        )
    print(f"wrote: teamocil layout: '{fname}'")

    # now start tmux session to run all jobs
    cmd = ["tmux", "new", "-d", "-s", name]

    exit_code = run_cmd(cmd)
    if exit_code != 0:
        print(f"failed to start tmux with code {exit_code}!")
        exit(exit_code)

    cmd = ["teamocil", "--layout", fname]
    exit_code = run_cmd(cmd)
    if exit_code != 0:
        print(f"failed to run teamocil with code {exit_code}!")
        exit(exit_code)

    print(
        f"\nsucessfully started tmux session '{name}' with {len(task_batch)} tasks running in {batch_num} parallel batches!"
    )
    print(f"\tyou can now join the tmux session to watch progress with:")
    print(f'\t\ttmux attach -t "{name}"')
    print(f"\tyou can kill the experiment anytime with:")
    print(f'\t\ttmux kill-session -t "{name}"')


if __name__ == "__main__":
    main()
