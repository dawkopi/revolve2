#!/usr/bin/env python3
"""Setup and running of the optimize modular program."""
import argparse
import glob
import logging
import subprocess
import numpy as np
import random

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen

import wandb

from optimizer import Optimizer as EaOptimzer
from optimizers.cma_optimizer import CmaEsOptimizer
from optimizers.ars_optimizer import ArsOptimizer
from utilities import *
from morphologies.morphology import MORPHOLOGIES
from genotypes.linear_controller_genotype import LinearControllerGenotype

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ERECTUS_YAML = os.path.join(SCRIPT_DIR, "morphologies/erectus.yaml")


async def main() -> None:
    """Run the optimization process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--run_name", type=str, default="default")
    parser.add_argument("--group_name", type=str, default="default")
    parser.add_argument("-l", "--resume_latest", action="store_true")
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("--rng_seed", type=int, default=None)
    parser.add_argument("--num_initial_mutations", type=int, default=10)
    parser.add_argument("-t", "--simulation_time", type=int, default=10)
    parser.add_argument("--sampling_frequency", type=float, default=10)
    parser.add_argument("--control_frequency", type=float, default=60)
    parser.add_argument(
        "-p",
        "--population_size",
        type=int,
        default=10,
        help="population size (but if using -ars this sets n_directions instead)",
    )
    parser.add_argument("-o", "--offspring_size", type=int, default=None)
    parser.add_argument("-g", "--num_generations", type=int, default=50)
    parser.add_argument(
        "-ms",
        "--max_steps",
        type=float,
        default=None,
        help="max steps (in millions e.g. 2.5)",
    )
    parser.add_argument("-w", "--wandb", action="store_true")
    parser.add_argument("--wandb_os_logs", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-cpu", "--n_jobs", type=int, default=1)
    parser.add_argument("-s", "--samples", type=int, default=4)
    parser.add_argument("--sigma0", type=float, default=0.2, help="param for CMA")
    parser.add_argument("--step_size", type=float, default=0.02, help="param for ARS")
    parser.add_argument(
        "-m",
        "--morphology",
        type=str,
        default="erectus",
        help="name of morphology to use (e.g. 'erecuts' | 'spider')",
    )
    parser.add_argument(
        "-f",
        "--fitness_function",
        # default="with_control_cost",
        # default="health_with_control_cost",
        default="clipped_health",
        # default="displacement_only",
    )  # "displacement_height_groundcontact"
    parser.add_argument(
        "--skip_best",
        action="store_true",
        help="don't output the best robots to disk",
    )
    parser.add_argument(
        "--best_dur",
        type=int,
        default=30,
        help="duration for rerun_best.py",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="run with non-headless mode (view sim window)",
    )
    parser.add_argument(
        "-cma",
        "--use_cma",
        action="store_true",
        help="use CMA-ES as optimizer of controller",
    )
    parser.add_argument(
        "-ars",
        "--use_ars",
        action="store_true",
        help="use ARS as optimizer of controller",
    )
    args = parser.parse_args()

    if args.rng_seed is None:
        args.rng_seed = random.randint(0, 999999)

    body_name = args.morphology
    assert body_name in MORPHOLOGIES, "morphology must exist"
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
        group=args.group_name,
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
    rng = random.Random()
    rng.seed(args.rng_seed)
    print(f"using random seed {args.rng_seed}")

    # process id generator
    process_id_gen = ProcessIdGen()
    process_id = process_id_gen.gen()

    ars_directions = None
    if args.use_cma:
        Optimizer = CmaEsOptimizer
        # logging.info(
        #     "CMA-ES start from an original individual, population size will be ignored."
        # )
        # args.population_size = 1

    elif args.use_ars:
        Optimizer = ArsOptimizer
        logging.info(
            "Ars start from an original individual, population size will be stay at 1"
        )
        ars_directions = int(args.population_size / 2)
        args.population_size = 1
        args.offspring_size = 1
    else:
        Optimizer = EaOptimzer

    logging.info(f"using optimizer: {Optimizer}")

    logging.info(f"using body_name: {body_name}")
    initial_population = [
        LinearControllerGenotype.random(body_name) for _ in range(args.population_size)
    ]

    # if args.use_cma:
    #     logging.info("CMA-ES self-adapt offspring size. (if not given)")
    #     args.population_size = 1
    #     if args.offspring_size is None:
    #         N = initial_population[0].genotype.shape[0]
    #         # self-adapted new generation size used in cma-es
    #         args.offspring_size = int(4 + 3 * np.log(N))

    # this if statement must be used after 'if args.use_cma:'
    if args.offspring_size is None:
        args.offspring_size = args.population_size

    # database
    database = open_async_database_sqlite(database_dir)
    maybe_optimizer = await Optimizer.from_database(
        database=database,
        process_id=process_id,
        innov_db_body=None,
        innov_db_brain=None,
        rng=rng,
        process_id_gen=process_id_gen,
        headless=not args.gui,
    )
    if maybe_optimizer is not None:
        logging.info(f"Initialized with existing database: '{database_dir}'")
        # TODO: if run is already finished, don't log it to wandb
        optimizer = maybe_optimizer
        optimizer._num_generations = (
            args.num_generations
        )  # in case more generations are desired :)
    else:
        logging.info(f"Initialized a new experiment: '{database_dir}'")
        optimizer = await Optimizer.new(
            database=database,
            process_id=process_id,
            initial_population=initial_population,
            rng=rng,
            process_id_gen=process_id_gen,
            innov_db_body=None,
            innov_db_brain=None,
            simulation_time=args.simulation_time,
            sampling_frequency=args.sampling_frequency,
            control_frequency=args.control_frequency,
            num_generations=args.num_generations,
            offspring_size=args.offspring_size,
            fitness_function=args.fitness_function,
            headless=not args.gui,
            body_name=body_name,
        )

    optimizer.n_jobs = args.n_jobs
    optimizer.samples = args.samples
    if isinstance(optimizer, ArsOptimizer):
        optimizer.override_params = {
            "n_directions": ars_directions,
            "deltas_used": ars_directions,
            "step_size": args.step_size,
            "seed": args.rng_seed,
        }
    if isinstance(optimizer, CmaEsOptimizer):
        optimizer.sigma0 = args.sigma0
    if args.max_steps is not None:
        args.max_steps = int(args.max_steps * 1_000_000)
        optimizer._max_sim_steps = args.max_steps
    max_steps_str = f"{args.max_steps:,}" if args.max_steps is not None else "None"

    logging.info(
        f"Starting optimization process (max generations={args.num_generations:,}, max steps={max_steps_str})..."
    )
    await optimizer.run()

    logging.info(
        f"Finished optimizing. (reached generation {optimizer.generation_index}/{args.num_generations}, sim step {optimizer._unique_sim_steps:,}/{max_steps_str})"
    )
    logging.info(f"database_dir = '{database_dir}'\n")

    if args.wandb:
        upload_db(database_dir)

    if not args.skip_best:
        logging.info("\n\nrunning rerun_best.py")
        call_rerun_best(run_name=args.run_name, count=4, dur_sec=args.best_dur)
        if args.wandb:
            # now save files to wandb if needed
            analysis_dir = os.path.join(database_dir, "analysis")
            UPLOAD_GLOBS = [
                os.path.join(analysis_dir, "*.webm"),
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


def upload_db(db_dir: str):
    """Compress db.sqlite -> dq.sqlite.tgz and upload to wandb."""
    tmp_file = os.path.join(db_dir, "db.sqlite.tgz")
    cmd = [
        "tar",
        "-cvzf",
        tmp_file,
        os.path.join(db_dir, "db.sqlite"),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    if res.returncode != 0:
        logging.error(f"rerun_best.py failed with code {res.returncode}")
    else:
        wandb.save(tmp_file)
        logging.info(f"uploaded compressed db to wandb: '{tmp_file}'")
        # os.remove(tmp_file) # causes wandb.save() to fail


def call_rerun_best(run_name: str, dur_sec: int = 30, count: int = 1):
    """Output video and xml files of best robots."""
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
        logging.error(f"rerun_best.py failed with code {res.returncode}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
