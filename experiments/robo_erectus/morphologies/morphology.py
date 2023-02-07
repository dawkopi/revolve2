#!/usr/bin/env python3
"""
Fixed morphology creator
refer to ci-group/revolve/experiments/examples/yaml & revolve/pyrevolve/revolve_bot/revolve_bot.py 
"""
import argparse
import os
import yaml
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.dirname(SCRIPT_DIR))
import math
from revolve2.core.modular_robot import ActiveHinge, Body, Core, Brick
from revolve2.core.physics.running._results import ActorState
from revolve2.core.physics.running import (
    Environment,
    PosedActor,
)
import logging
from revolve2.runners.mujoco import LocalRunner

import utilities
from utilities import (
    actor_get_default_pose,
    actor_get_standing_pose,
)

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


class FixedBodyCreator:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.load_file(yaml_file)

    def load_file(self, path, conf_type="yaml"):
        """
        Read robot's description from a file and parse it to Python structure
        :param path: Robot's description file path
        :param conf_type: Type of a robot's description format
        :return:
        """
        with open(path, "r") as robot_file:
            text = robot_file.read()

        if "yaml" == conf_type:
            self.load_yaml(text)
        elif "sdf" == conf_type:
            raise NotImplementedError("Loading from SDF not yet implemented")

    def load_yaml(self, text):
        """
        Load robot's description from a yaml string
        :param text: Robot's yaml description
        """
        yaml_bot = yaml.safe_load(text)
        self._id = yaml_bot["id"] if "id" in yaml_bot else None
        self._core = self.FromYaml(yaml_bot["body"])
        self._body = Body()
        self._body.core = self._core
        self._body.finalize()

    def FromYaml(self, yaml_object):
        """
        From a yaml object, creates a data struture of interconnected body modules.
        Standard names for modules are:
        Core
        ActiveHinge
        Brick
        """
        mod_type = yaml_object["type"]

        try:
            rotation = yaml_object["orientation"] / 180 * math.pi
        except KeyError:
            rotation = 0.0

        if mod_type == "CoreComponent" or mod_type == "Core":
            module = Core(rotation)
        elif mod_type == "ActiveHinge":
            module = ActiveHinge(rotation)
        elif mod_type == "Brick":
            module = Brick(rotation)
        else:
            raise NotImplementedError(
                '"{}" module not yet implemented'.format(mod_type)
            )

        # module.id = yaml_object['id']

        if "children" in yaml_object:
            for parent_slot in yaml_object["children"]:
                module.children[parent_slot] = self.FromYaml(
                    yaml_object=yaml_object["children"][parent_slot]
                )

        return module

    @property
    def body(self):
        return self._body


# defines all hardcoded morphologies for a genotype's (hyper) params
# each key corresponds to the yaml file "./{key}.yaml"
MORPHOLOGIES = {
    "erectus": {
        "min_z": 0.4,  # for health check
        "get_pose": actor_get_standing_pose,
    },
    "erectus_rot": {  # same as erectus but knee rotated to bend in direction a human's would:
        "min_z": 0.4,  # for health check
        "get_pose": actor_get_standing_pose,
    },
    "dangle": {
        "min_z": 0.13,  # for health check
        "get_pose": actor_get_standing_pose,
    },
    "trirectus": {
        "min_z": 0.1,  # for health check
        "get_pose": actor_get_default_pose,
    },
    "spider": {
        "min_z": 0.1,
        "get_pose": actor_get_default_pose,
    },
    "dog": {
        "min_z": 0.22,
        "get_pose": actor_get_default_pose,
    },
    "humanoid": {
        "min_z": 0.45,
        "get_pose": actor_get_standing_pose,
    },
    "humanoid_B": {  # shorter limbs
        "min_z": 0.34,
        "get_pose": actor_get_standing_pose,
    },
    "humanoid_C": {  # rotated body
        "min_z": 0.45,
        "get_pose": actor_get_standing_pose,
    },
    "humanoid_D": {  # asymmetric
        "min_z": 0.45,
        "get_pose": actor_get_standing_pose,
    },
    "humanoid_E": {  # longer thigh, shorter calf
        "min_z": 0.45,
        "get_pose": actor_get_standing_pose,
    },
    "humanoid_F": {  # longer legs
        "min_z": 0.1,
        "get_pose": actor_get_standing_pose,
    },
    "humanoid_G": {  # different angle of joins
        "min_z": 0.5,
        "get_pose": actor_get_standing_pose,
    },
    "humanoid_H": {  # different angle of joins
        "min_z": 0.5,
        "get_pose": actor_get_standing_pose,
    },
    "humanoid_I": {  # different angle of joins
        "min_z": 0.5,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_000": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_001": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_010": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_011": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_100": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_101": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_110": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_111": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_01-0": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_01-1": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_10-0": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_10-1": {
        "min_z": 0.3,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_0-01": {
        "min_z": 0.35,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_0-10": {
        "min_z": 0.35,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_1-01": {
        "min_z": 0.35,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_1-10": {
        "min_z": 0.35,
        "get_pose": actor_get_standing_pose,
    },
    "erectus_10-10-10": {
        "min_z": 0.38,
        "get_pose": actor_get_standing_pose,
    },
}

for key, m in MORPHOLOGIES.items():
    fname = os.path.join(SCRIPT_DIR, key + ".yaml")
    if not os.path.isfile(fname):
        raise FileNotFoundError(fname + " not found")
    m["fname"] = fname

    min_z = m["min_z"]
    # if "min_z" not in m:
    # logging.warn(f"body '{key}' has no min_z --- (its health checking is disabled)")

    def healthy_factory(
        min_z,
    ):  # this extra func prevents annoying memory issue with min_z
        def is_healthy(state: ActorState):
            return utilities.is_healthy_state(state, min_z=min_z)

        return is_healthy

    # m["is_healthy"] = lambda state: utilities.is_healthy_state(state, min_z)
    m["is_healthy"] = healthy_factory(min_z)


def output_all():
    for body_name in MORPHOLOGIES.keys():
        output_morphology(body_name)


def output_morphology(body_name: str):
    assert body_name in MORPHOLOGIES, "morphology must exist"

    from genotypes.linear_controller_genotype import LinearControllerGenotype

    genotype = LinearControllerGenotype.random(body_name)
    actor, controller = genotype.develop()

    pos, rot = genotype.get_initial_pose(actor)
    env = Environment()
    env.actors.append(
        PosedActor(
            actor,
            pos,
            rot,
            [0.0 for _ in controller.get_dof_targets()],
        ),
    )

    outpath = os.path.join(SCRIPT_DIR, f"{body_name}.xml")
    xml_string = LocalRunner._make_mjcf(env)
    with open(outpath, "w") as f:
        f.write(xml_string)
    print(f"wrote '{body_name}' body to: '{outpath}'")

    with open(os.path.join(SCRIPT_DIR, f"cur.xml"), "w") as f:
        f.write(xml_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="output a given morphology to xml (to visualize with mujoc 'simulate' program)"
    )
    parser.add_argument(
        "-m",
        "--morphology",
        type=str,
        default=None,
        help="name of morphology to use (e.g. 'erecuts' | 'spider')",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list all morphologies",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="output all morphologies to xml files",
    )
    args = parser.parse_args()

    if args.list:
        print("list of all morphologies:")
        print("\n".join(MORPHOLOGIES.keys()))
        print()
    if args.all:
        output_all()
    elif args.morphology:
        body_name = args.morphology
        output_morphology(body_name)
        # erectus_000,erectus_10-10-10,humanoid_I
    else:
        print("no morphology selected!\n")
        print(parser.format_help())
        exit(1)
