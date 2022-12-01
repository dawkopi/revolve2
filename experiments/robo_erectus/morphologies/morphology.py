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
        if mod_type == "CoreComponent" or mod_type == "Core":
            module = Core(0.0)
        elif mod_type == "ActiveHinge":
            module = ActiveHinge(math.pi / 2.0)
        elif mod_type == "Brick":
            module = Brick(0.0)
        else:
            raise NotImplementedError(
                '"{}" module not yet implemented'.format(mod_type)
            )

        # module.id = yaml_object['id']

        try:
            module.orientation = yaml_object["orientation"]
        except KeyError:
            module.orientation = 0

        try:
            module.rgb = (
                yaml_object["params"]["red"],
                yaml_object["params"]["green"],
                yaml_object["params"]["blue"],
            )
        except KeyError:
            pass

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
        "min_z": 0.1,  # for health check
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
    "birectus_simple": {
        "min_z": 0.1,  # for health check
        "get_pose": actor_get_default_pose,
    },
    "spider": {
        "min_z": 0.1,
        "get_pose": actor_get_default_pose,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="output a given morphology to xml (to visualize with mujoc 'simulate' program)"
    )
    parser.add_argument(
        "-m",
        "--morphology",
        type=str,
        default="erectus",
        help="name of morphology to use (e.g. 'erecuts' | 'spider')",
    )
    args = parser.parse_args()

    body_name = args.morphology
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
