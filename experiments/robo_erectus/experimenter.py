#!/usr/bin/env python3
import argparse
import os
import subprocess
import copy
from typing import List
import yaml
import math
from multiprocessing import cpu_count
from morphologies.morphology import MORPHOLOGIES

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
OPTIMIZE = os.path.join(SCRIPT_DIR, "optimize.py")
# OPTIMIZE = "./optimize.py"
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def main():
    CPUS = cpu_count()
    parser = argparse.ArgumentParser(
        description=f"Runs a set of experiments by calling {os.path.basename(OPTIMIZE)}"
    )
    # erectus_000
    parser.add_argument(
        "-m",
        "--morphology",
        type=str,
        required=True,
        help="morphology names, separated by commas if multiple",
    )
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
        default=max(1, int(CPUS / 2)),
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
    parser.add_argument(
        "--venv",
        type=str,
        default=os.path.join(ROOT_DIR, ".venv"),
        help="path to virtualenv environment dir for tasks to use",
    )
    # parser.add_argument("-t", "--simulation_time", type=int, default=10)
    args = parser.parse_args()

    wandb_flags = []
    if not args.no_wandb:
        wandb_flags = ["-w", "--wandb_os_logs"]

    assert os.path.exists(OPTIMIZE)
    assert os.path.isdir(args.venv), f"virtualenv dir not found: '{args.venv}'"

    task_batch = []  # build list of commands to run
    morphologies = args.morphology.split(",")
    for morphology in morphologies:
        assert morphology in MORPHOLOGIES, f"morphology must exist: '{morphology}'"
        for opti in ["cma", "ars"]:
            for pop in [10, 100]:
                # NOTE: -cpu flag gets added in schedule_jobs()
                base_cmd = [
                    "time",
                    "python3",
                    OPTIMIZE,
                    "-m",
                    morphology,
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
                        group_name = f"{args.prefix + '_' if args.prefix else ''}_{morphology}_{opti}_pop{pop}_sigma{sigma0}"
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
                        group_name = f"{args.prefix + '_' if args.prefix else ''}_{morphology}_{opti}_pop{pop}_step_size{step_size}"
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
    print("printing commands for reference: (-cpu flag will be added later)")
    for i, cmd in enumerate(task_batch):
        # run_cmd(cmd, dry_run=args.dry, msg=f"{i+1}/{len(task_batch)}")
        # run_cmd(cmd, dry_run=True, msg=f"{i+1}/{len(task_batch)}")
        # here we just print the commands for reference:
        # print(f"command {str(i+1).zfill(2)}/{len(task_batch)}:\t" + " ".join(cmd))
        print(" ".join(cmd))

    args.max_cpus = min(CPUS, args.max_cpus)
    schedule_jobs(
        task_batch,
        args.prefix,
        args.max_cpus,
        args.venv,
        dry_run=args.dry,
    )


def schedule_jobs(
    task_batch: List,
    name: str,
    max_cpus: int,
    venv_dir: str,
    dry_run=False,
):
    """
    Schedule jobs to run in a tmux session.
    Requires teamocil and tmux to be installed.  https://github.com/remi/teamocil#installation
    params:
        task_batch: list of commands (where each command is a list strings)
        max_cpus: max number of tasks to run in parallel (as separate tmux tabs)
        name: name of tmux session to create
    """
    num_tasks = len(task_batch)
    print(
        f"\nscheduling {num_tasks} tasks, total cpu budget is {max_cpus} (of {cpu_count()} available)..."
    )
    tasks = copy.deepcopy(task_batch)

    # decide scheduling / calculate cpus_per_task
    num_batches = max(1, min(8, num_tasks, max_cpus))  # (aim for up to 8 batches)
    remaining_cpus = max_cpus
    # bin_size = math.ceil(num_tasks / num_batches)

    precommands = [f"source '{os.path.join(venv_dir, 'bin/activate')}'"]
    # create dict defining layout of tmux session (for teamocil)
    data = {
        "name": name,
        "windows": [],
    }
    for b in range(num_batches):
        remaining_batches = num_batches - (b)  # counting this one
        if b != num_batches - 1:
            batch_cpus = max(1, int(remaining_cpus / remaining_batches))
            # batch_cpus = cpus_per_task
            # batch_size = max(1, round(num_tasks * (batch_cpus / max_cpus)))
            batch_size = max(1, round(len(tasks) * (batch_cpus / remaining_cpus)))
        else:
            # last batch may get a higher proportion of tasks (as it may have more cpus)
            batch_size = len(tasks)
            batch_cpus = remaining_cpus

        remaining_cpus -= batch_cpus
        assert batch_cpus >= 1

        # add -cpu flag to all tasks in batch
        cur_tasks = [[*cmd, "-cpu", str(batch_cpus)] for cmd in tasks[0:batch_size]]
        # flatten tasks into list of strings
        cur_tasks = [" ".join(cmd) for cmd in cur_tasks]
        batch_name = f"batch{str(b+1).zfill(2)}"
        data["windows"].append(
            {
                "name": batch_name,
                "root": SCRIPT_DIR,
                "layout": "tiled",
                # "panes": ["date; " + "; date; ".join(cur_tasks)], # chain commands together to run in series
                "panes": [{"commands": [*precommands, *cur_tasks]}],
            }
        )
        print(f"{batch_name}: {len(cur_tasks)} tasks,\t{batch_cpus} cpus")
        tasks = tasks[batch_size:]
    assert len(tasks) == 0 and remaining_cpus == 0

    fname = os.path.join(SCRIPT_DIR, f"layout_{name}.yml")
    with open(fname, "w") as f:
        yaml.dump(
            data,
            f,
        )
    print(f"\nwrote: teamocil layout: '{fname}'")
    if dry_run:
        print(f"not starting tmux session due to dry run mode!")
        return

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
        f"\nsucessfully started tmux session '{name}' with {num_tasks} tasks running in {num_batches} parallel batches!"
    )
    print(f"\tyou can kill the experiment anytime with:")
    print(f'\t\ttmux kill-session -t "{name}"')
    print(f"\tyou can join the tmux session to watch progress with:")
    print(f'\t\ttmux attach -t "{name}"')


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


if __name__ == "__main__":
    main()
