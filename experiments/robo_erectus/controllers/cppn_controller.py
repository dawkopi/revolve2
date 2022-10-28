import math
from typing import List, Tuple, cast, Set

import multineat
from revolve2.core.modular_robot import ActiveHinge, Body, Brain
from revolve2.actor_controller import ActorController
from revolve2.actor_controllers.cpg import (
    CpgNetworkStructure,
    CpgPair,
    CpgActorController as ControllerCpg,
)


class CppnController(Brain):
    """
    A CPG brain based on `ModularRobotBrainCpgNetworkNeighbour` that creates weights from a CPPNWIN network.

    Weights are determined by querying the CPPN network with inputs:
    (hinge1_posx, hinge1_posy, hinge1_posz, hinge2_posx, hinge2_posy, hinge3_posz)
    If the weight in internal, hinge1 and hinge2 position will be the same.
    """

    _genotype: multineat.Genome

    def __init__(self, genotype: multineat.Genome):
        """
        Initialize this object.

        :param genotype: A multineat genome used for determining weights.
        """
        self._genotype = genotype

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        """
        Create a controller for the provided body.

        :param body: The body to make the brain for.
        :param dof_ids: Map from actor joint index to module id.
        :returns: The created controller.
        """
        # get active hinges and sort them according to dof_ids
        active_hinges_unsorted = body.find_active_hinges()
        active_hinge_map = {
            active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
        }
        active_hinges = [active_hinge_map[id] for id in dof_ids]

        cpg_network_structure = self._make_cpg_network_structure_neighbour(
            active_hinges
        )
        connections = [
            (
                active_hinges[pair.cpg_index_lowest.index],
                active_hinges[pair.cpg_index_highest.index],
            )
            for pair in cpg_network_structure.connections
        ]

        (internal_weights, external_weights) = self._make_weights(
            active_hinges, connections, body
        )
        weight_matrix = cpg_network_structure.make_connection_weights_matrix(
            {
                cpg: weight
                for cpg, weight in zip(cpg_network_structure.cpgs, internal_weights)
            },
            {
                pair: weight
                for pair, weight in zip(
                    cpg_network_structure.connections, external_weights
                )
            },
        )
        initial_state = cpg_network_structure.make_uniform_state(0.5 * math.sqrt(2))
        dof_ranges = cpg_network_structure.make_uniform_dof_ranges(1.0)

        return ControllerCpg(
            initial_state, cpg_network_structure.num_cpgs, weight_matrix, dof_ranges
        )

    def _make_cpg_network_structure_neighbour(
        self,
        active_hinges: List[ActiveHinge],
    ) -> CpgNetworkStructure:
        """
        Create the structure of a cpg network based on a list of active hinges.

        The order of the active hinges matches the order of the cpgs.
        I.e. every active hinges has a corresponding cpg,
        and these are stored in the order the hinges are provided in.

        :param active_hinges: The active hinges to base the structure on.
        :returns: The created structure.
        """
        cpgs = CpgNetworkStructure.make_cpgs(len(active_hinges))
        connections: Set[CpgPair] = set()

        active_hinge_to_cpg = {
            active_hinge: cpg for active_hinge, cpg in zip(active_hinges, cpgs)
        }

        for active_hinge, cpg in zip(active_hinges, cpgs):
            neighbours = [
                n
                for n in active_hinge.neighbours(within_range=2)
                if isinstance(n, ActiveHinge)
            ]
            connections = connections.union(
                [
                    CpgPair(cpg, active_hinge_to_cpg[neighbour])
                    for neighbour in neighbours
                ]
            )

        return CpgNetworkStructure(cpgs, connections)

    def _make_weights(
        self,
        active_hinges: List[ActiveHinge],
        connections: List[Tuple[ActiveHinge, ActiveHinge]],
        body: Body,
    ) -> Tuple[List[float], List[float]]:
        brain_net = multineat.NeuralNetwork()
        self._genotype.BuildPhenotype(brain_net)

        internal_weights = [
            self._evaluate_network(
                brain_net,
                [
                    1.0,
                    float(pos.x),
                    float(pos.y),
                    float(pos.z),
                    float(pos.x),
                    float(pos.y),
                    float(pos.z),
                ],
            )
            for pos in [
                body.grid_position(active_hinge) for active_hinge in active_hinges
            ]
        ]

        external_weights = [
            self._evaluate_network(
                brain_net,
                [
                    1.0,
                    float(pos1.x),
                    float(pos1.y),
                    float(pos1.z),
                    float(pos2.x),
                    float(pos2.y),
                    float(pos2.z),
                ],
            )
            for (pos1, pos2) in [
                (body.grid_position(active_hinge1), body.grid_position(active_hinge2))
                for (active_hinge1, active_hinge2) in connections
            ]
        ]

        return (internal_weights, external_weights)

    @staticmethod
    def _evaluate_network(
        network: multineat.NeuralNetwork, inputs: List[float]
    ) -> float:
        network.Input(inputs)
        network.ActivateAllLayers()
        return cast(float, network.Output()[0])  # TODO missing multineat typing
