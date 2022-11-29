import os
import numpy as np
import math
import sqlalchemy
from typing import List, Optional
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from controllers.linear_controller import LinearController
from revolve2.core.database import IncompatibleError, Serializer
from revolve2.core.physics.actor import Actor
from morphologies.morphology import FixedBodyCreator, MORPHOLOGIES


class LinearControllerGenotype:
    def __init__(self, genotype, body_name: str):
        self.genotype = genotype
        self.body_name = body_name  # for morphology
        self.is_healthy = MORPHOLOGIES[body_name]["is_healthy"]

    def develop(self):
        actor, dof_ids = LinearControllerGenotype.develop_body(self.body_name)

        dof_size = len(dof_ids)
        input_size = LinearController.get_input_size(dof_size)
        policy = self.genotype.reshape((input_size, dof_size))
        controller = LinearController(policy)

        return actor, controller

    def get_initial_pose(self, actor: Actor):
        return MORPHOLOGIES[self.body_name]["get_pose"](actor)

    @classmethod
    def random(cls, body_name: str):
        _, dof_ids = LinearControllerGenotype.develop_body(body_name)
        dof_size = len(dof_ids)
        input_size = LinearController.get_input_size(dof_size)
        genotype = np.random.normal(scale=0.1, size=(input_size, dof_size)).flatten()
        return LinearControllerGenotype(genotype, body_name)

    @staticmethod
    def develop_body(body_name: str):
        if body_name not in MORPHOLOGIES:
            raise KeyError(f"body_name '{body_name}' not found in MORPHOLOGIES")

        body_info = MORPHOLOGIES[body_name]
        assert os.path.exists(body_info["fname"])
        body = FixedBodyCreator(body_info["fname"]).body
        actor, dof_ids = body.to_actor()
        return actor, dof_ids

        # fallback:
        # Hardcoded body; for now
        body = Body()
        body.core.front = ActiveHinge(0)
        body.core.front._absolute_rotation = 0
        body.core.front.attachment = ActiveHinge(math.pi / 2.0)
        body.core.front.attachment._absolute_rotation = 0
        body.core.left = ActiveHinge(0)
        body.core.left._absolute_rotation = 0
        body.core.left.attachment = ActiveHinge(math.pi / 2.0)
        body.core.left.attachment._absolute_rotation = 0
        body.core.right = ActiveHinge(0)
        body.core.right._absolute_rotation = 0
        body.core.right.attachment = ActiveHinge(math.pi / 2.0)
        body.core.right.attachment._absolute_rotation = 0
        body.core.back = ActiveHinge(0)
        body.core.back._absolute_rotation = 0
        body.core.back.attachment = ActiveHinge(math.pi / 2.0)
        body.core.back.attachment._absolute_rotation = 0
        body.finalize()
        actor, dof_ids = body.to_actor()
        return actor, dof_ids


DbBase = declarative_base()


class DbGenotype(DbBase):
    """Stores serialized multineat genomes."""

    __tablename__ = "cppnwin_genotype"

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )
    serialized_genome = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    body_name = sqlalchemy.Column(
        sqlalchemy.String, nullable=False
    )  # e.g. "erectus" | "spider"


class LinearGenotypeSerializer(Serializer[LinearControllerGenotype]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

    @classmethod
    def identifying_table(cls) -> str:
        return "linear_genotype"

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[LinearControllerGenotype]
    ) -> List[int]:
        dbfitnesses = [
            DbGenotype(
                serialized_genome=np.array(o.genotype, dtype=float).tostring(),
                body_name=o.body_name,
            )
            for o in objects
        ]
        session.add_all(dbfitnesses)
        await session.flush()
        ids = [dbfitness.id for dbfitness in dbfitnesses if dbfitness.id is not None]
        assert len(ids) == len(objects)
        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[LinearControllerGenotype]:
        rows = (
            (await session.execute(select(DbGenotype).filter(DbGenotype.id.in_(ids))))
            .scalars()
            .all()
        )

        if len(rows) != len(ids):
            raise IncompatibleError()

        id_map = {t.id: t for t in rows}
        genotypes = [
            LinearControllerGenotype(
                np.fromstring(id_map[id].serialized_genome, dtype=float),
                id_map[id].body_name,
            )
            for id in ids
        ]
        return genotypes
