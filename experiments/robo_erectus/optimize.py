#!/usr/bin/env python3
"""Setup and running of the optimize modular program."""
import argparse
import glob
import logging
import subprocess
import numpy as np
from random import Random

import multineat
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen

import wandb

from optimizer import Optimizer as EaOptimzer
from optimizers.cma_optimizer import CmaEsOptimizer
from utilities import *
from genotypes.linear_controller_genotype import LinearControllerGenotype

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ERECTUS_YAML = os.path.join(SCRIPT_DIR, "morphologies/erectus.yaml")


async def main() -> None:
    """Run the optimization process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--run_name", type=str, default="default")
    parser.add_argument("-l", "--resume_latest", action="store_true")
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("--rng_seed", type=int, default=420)
    parser.add_argument("--num_initial_mutations", type=int, default=10)
    parser.add_argument("-t", "--simulation_time", type=int, default=30)
    parser.add_argument("--sampling_frequency", type=float, default=10)
    parser.add_argument("--control_frequency", type=float, default=60)
    parser.add_argument("-p", "--population_size", type=int, default=10)
    parser.add_argument("--offspring_size", type=int, default=None)
    parser.add_argument("-g", "--num_generations", type=int, default=50)
    parser.add_argument("-w", "--wandb", action="store_true")
    parser.add_argument("--wandb_os_logs", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-cpu", "--n_jobs", type=int, default=1)
    parser.add_argument(
        "-m",
        "--morphology",
        type=str,
        default=ERECTUS_YAML,
        help="yaml file to use for robot's morphology",
    )
    parser.add_argument(
        "-f",
        "--fitness_function",
        # default="with_control_cost",
        default="health_with_control_cost",
    )  # "displacement_height_groundcontact"
    parser.add_argument(
        "-b",
        "--save_best",
        action="store_true",
        help="output the best robots to disk",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="run with non-headless mode (view sim window)",
    )
    parser.add_argument(
        "--use_cma",
        action="store_true",
        help="use CMA-ES as optimizer of controller",
    )
    args = parser.parse_args()

    body_yaml = args.morphology
    assert os.path.exists(body_yaml)
    ensure_dirs(DATABASE_PATH)

    # https://docs.wandb.ai/guides/track/advanced/resuming#resuming-guidance
    should_resume = args.resume_latest or args.resume
    full_run_name = None
    if args.resume_latest:
        full_run_name = get_latest_run()
    elif args.resume:
        full_run_name = find_dir(DATABASE_PATH, args.run_name)

    if should_resume and full_run_name is None:
        logging.error("Run not found...")
        exit()

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="robo-erectus",
        entity="ea-research",
        config=vars(args),
        settings=wandb.Settings(
            _disable_stats=not args.wandb_os_logs,
            _disable_meta=not args.wandb_os_logs,
        ),
        resume=should_resume,
    )

    if full_run_name is None:
        full_run_name = f"{args.run_name}__{wandb.run.name}"

    database_dir = os.path.join(DATABASE_PATH, full_run_name)
    wandb.run.name = full_run_name
    set_latest_run(full_run_name)

    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # random number generator
    rng = Random()
    rng.seed(args.rng_seed)

    # database
    database = open_async_database_sqlite(database_dir)

    # process id generator
    process_id_gen = ProcessIdGen()
    process_id = process_id_gen.gen()

    # multineat innovation databases
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    if args.use_cma:
        Optimizer = CmaEsOptimizer
        logging.info(
            "CMA-ES start from an original individual, population size will be ignored."
        )
        args.population_size = 1

    initial_population = [
        LinearControllerGenotype.random(body_yaml) for _ in range(args.population_size)
    ]

    if args.use_cma:
        logging.info("CMA-ES self-adapt offspring size. (if not given)")
        args.population_size = 1
        if args.offspring_size is None:
            N = initial_population[0].genotype.shape[0]
            # self-adapted new generation size used in cma-es
            args.offspring_size = int(4 + 3 * np.log(N))
    else:
        Optimizer = EaOptimzer

    # this if statement must be used after 'if args.use_cma:'
    if args.offspring_size is None:
        args.offspring_size = args.population_size

    maybe_optimizer = await Optimizer.from_database(
        database=database,
        process_id=process_id,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
        process_id_gen=process_id_gen,
        headless=not args.gui,
    )
    if maybe_optimizer is not None:
        logging.info(f"Initialized with existing database: '{database_dir}'")
        # TODO: if run is already finished, don't log it to wandb
        optimizer = maybe_optimizer
    else:
        logging.info(f"Initialized a new experiment: '{database_dir}'")
        optimizer = await Optimizer.new(
            database=database,
            process_id=process_id,
            initial_population=initial_population,
            rng=rng,
            process_id_gen=process_id_gen,
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            simulation_time=args.simulation_time,
            sampling_frequency=args.sampling_frequency,
            control_frequency=args.control_frequency,
            num_generations=args.num_generations,
            offspring_size=args.offspring_size,
            fitness_function=args.fitness_function,
            headless=not args.gui,
            body_yaml=body_yaml,
        )

    logging.info("Starting optimization process...")

    optimizer.n_jobs = args.n_jobs
    optimizer.body_yaml = body_yaml
    await optimizer.run()

    logging.info("Finished optimizing.")

    if args.save_best:
        logging.info("\n\nrunning rerun_best.py")
        call_rerun_best(run_name=args.run_name, count=4)
        if args.wandb:
            # now save files wandb if needed
            analysis_dir = os.path.join(database_dir, "analysis")
            UPLOAD_GLOBS = [
                os.path.join(analysis_dir, "*.mp4"),
                os.path.join(analysis_dir, "*.yml"),
            ]
            fnames = []
            for pattern in UPLOAD_GLOBS:
                for fname in glob.glob(pattern):
                    fnames.append(fname)
                    # https://docs.wandb.ai/guides/track/advanced/save-restore
                    # logging.debug(f"uploading: {fname}")
            logging.info(f"found {len(fnames)} files to upload to wandb...")
            [wandb.save(fname) for fname in fnames]

    if args.wandb:
        # TODO: upload compressed database!
        pass


def call_rerun_best(run_name: str, dur_sec: int = 30, count: int = 1):
    """Output mp4 and xml files of best robots."""
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
    RERUN_SCRIPT = os.path.join(SCRIPT_DIR, "rerun_best.py")

    cmd = [
        "python3",
        RERUN_SCRIPT,
        "-n",
        run_name,
        "-t",
        str(dur_sec),
        "-c",
        str(count),
        "--video",
    ]
    logging.debug("running command:")
    logging.debug(" ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    if res.returncode != 0:
        logging.error(f"rerun_best.py failed with code {res.return_code}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
