#!/usr/bin/env python3
import argparse
import os
import subprocess
import copy

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

OPTIMIZE = os.path.join(SCRIPT_DIR, "optimize.py")


def main():
    parser = argparse.ArgumentParser()
    # erectus_000
    parser.add_argument("-m", "--morphology", type=str, default="erectus_000")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="prefix for run names (sets group_name in wandb)",
    )
    parser.add_argument("--dry", action="store_true", default=False)
    parser.add_argument(
        "-nw", "--no-wandb", action="store_true", default=False, help="don't use wandb"
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
        wandb_flags = ["-w", "--wandb_os_logs", "--group_name", args.prefix]

    assert os.path.exists(OPTIMIZE)

    task_batch = []  # build list of commands to run
    for opti in ["cma", "ars"]:
        for pop in [10, 100]:
            base_cmd = [
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
                    for trial in range(args.trials):
                        run_name = f"{args.prefix + '_' if args.prefix else ''}_{opti}_pop{pop}_sigma{sigma0}_trial{trial}"
                        cmd = [
                            *base_cmd,
                            "-cma",
                            "--sigma0",
                            str(sigma0),
                            "-n",
                            run_name,
                        ]
            elif opti == "ars":
                run_name = f"{args.prefix + '_' if args.prefix else ''}{opti}pop{pop}"
                for step_size in [0.002, 0.02, 0.2]:
                    for trial in range(args.trials):
                        run_name = f"{args.prefix + '_' if args.prefix else ''}_{opti}_pop{pop}_sigma{sigma0}_trial{trial}"

                        cmd = [
                            *base_cmd,
                            "-ars",
                            "--step_size",
                            str(step_size),
                            "-n",
                            run_name,
                        ]
                        task_batch.append(cmd)

            else:
                raise NotImplementedError

    print(
        f"collected {len(task_batch)} commands to {'dry' if args.dry_run else ''} run\n"
    )
    # TODO: implement some way of scheduling jobs, and use -cpu flag!!
    for i, cmd in enumerate(task_batch):
        run_cmd(cmd, dry_run=args.dry, msg=f"{i+1}/{len(task_batch)}")


exit_codes = []


def run_cmd(cmd, msg="", dry_run=False):
    print(f"\n{'running' if not dry_run else 'would run'} command {msg}:")
    print(" ".join(cmd))
    if dry_run:
        return

    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    exit_codes.append(res.returncode)
    if res.returncode != 0:
        print(f"***command failed with code {res.returncode}")
        print(" ".join(cmd))


# pop_sizes=(10 50 100)
# sigmas=(0.2 0.5 0.8)
#
# for pop_size in ${pop_sizes[@]}
# do
# {
#   for sigma in ${sigmas[@]}
#   do
#   {
#          echo  "$pop_size, $sigma"
#
#          python experiments/robo_erectus/optimize.py -n "${pop_size}p_4s_${sigma}sigma_erectus_000" -m erectus_000 -cma -p $pop_size --sigma0 $sigma -s 4 -cpu 2 -g 300 -w --wandb_os_logs &       #循环内容放到后台执行
#   }
#   done
# }
# done
#
# wait      #等待循环结束再执行wait后面的内容
#
# echo -e "time-consuming: $SECONDS    seconds"    #显示脚本执行耗时

if __name__ == "__main__":
    main()
