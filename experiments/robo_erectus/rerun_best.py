#!/usr/bin/env python3
"""Visualize and simulate the best robot from the optimization process."""

import argparse
import os

# from optimizer import actor_get_standing_pose, actor_get_default_pose
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
from revolve2.runners.mujoco import LocalRunner, ModularRobotRerunner
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select


# from optimize import ERECTUS_YAML
from utilities import *

from genotypes.linear_controller_genotype import (
    LinearControllerGenotype,
    LinearGenotypeSerializer,
)

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
# ERECTUS_YAML = os.path.join(SCRIPT_DIR, "morphologies/erectus.yaml")


async def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="reruns simulation for all time best robot"
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=1000000,
        help="time (secs) for which to run the simulation",
    )
    parser.add_argument("-l", "--load_latest", action="store_true")
    parser.add_argument("-n", "--run_name", type=str, default="default")
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=1,
        help="quantity of 'best' robots to display in order",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="whether to write video of sim to file (runs headless)",
    )
    args = parser.parse_args()
    ensure_dirs(DATABASE_PATH)

    if args.load_latest:
        full_run_name = get_latest_run()
    else:
        full_run_name = find_dir(DATABASE_PATH, args.run_name)

    if full_run_name is None:
        print("Run not found...")
        exit()
    else:
        print(f'Run found - "{full_run_name}"')
    if args.count > 1 and args.time > 300:
        print(
            "WARNING: consider using a shorter simulation time (-t) when visualizing multiple robots\n"
        )

    database_dir = os.path.join(DATABASE_PATH, full_run_name)
    analysis_dir = os.path.join(database_dir, ANALYSIS_DIR_NAME)
    ensure_dirs(analysis_dir)

    db = open_async_database_sqlite(database_dir)
    best_individuals = []
    max_count = args.count
    async with AsyncSession(db) as session:
        best_individuals = (
            await session.execute(
                select(DbEAOptimizerIndividual, DbFloat)
                .filter(DbEAOptimizerIndividual.fitness_id == DbFloat.id)
                .order_by(DbFloat.value.desc())
                .limit(max_count)
            )
        ).all()
        print(f"found {len(best_individuals)} best robots\n")

        for i in range(len(best_individuals)):
            res = best_individuals[i]
            genotype = (
                await LinearGenotypeSerializer.from_database(
                    session, [res[0].genotype_id]
                )
            )[0]
            ind_id = res[0].individual_id
            print(
                f"rank: {i}, individual_id: {ind_id}, genotype_id: {res[0].genotype_id}, fitness: {res[1].value:0.5f}"
            )

            rerunner = ModularRobotRerunner()

            pose_getter = genotype.get_initial_pose
            actor, controller = genotype.develop()
            env, _ = ModularRobotRerunner.robot_to_env(actor, controller, pose_getter)

            # output env to a MJCF (xml) file (based on LocalRunner.run_batch())
            xml_string = LocalRunner._make_mjcf(env)
            # model = mujoco.MjModel.from_xml_string(xml_string)
            # data = mujoco.MjData(model)

            fname_base = f"rank{str(i).zfill(3)}_ind_id{ind_id}"
            xml_path = os.path.join(analysis_dir, f"{fname_base}.xml")
            with open(xml_path, "w") as f:
                f.write(xml_string)
            print(f"wrote file: '{xml_path}'")
            video_path = (
                os.path.join(analysis_dir, f"{fname_base}.mp4") if args.video else ""
            )

            # run simulation
            print(f"starting simulation for {args.time} secs...")
            await rerunner.rerun(
                actor,
                controller,
                60,
                simulation_time=args.time,
                get_pose=pose_getter,
                video_path=video_path,
            )
            if video_path:
                print(f"wrote file: {video_path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
