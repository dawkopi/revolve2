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
from morphologies.morphology import FixedBodyCreator


class LinearControllerGenotype:
    def __init__(self, genotype, yaml_file: str):
        self.genotype = genotype
        self.yaml_file = yaml_file  # for morphology

    def develop(self):
        actor, dof_ids = LinearControllerGenotype.develop_body(self.yaml_file)

        dof_size = len(dof_ids)
        input_size = LinearController.get_input_size(dof_size)
        policy = self.genotype.reshape((input_size, dof_size))
        controller = LinearController(policy)

        return actor, controller

    @classmethod
    def random(cls, yaml_file: Optional[str]):
        _, dof_ids = LinearControllerGenotype.develop_body(yaml_file)
        dof_size = len(dof_ids)
        input_size = LinearController.get_input_size(dof_size)
        genotype = np.random.normal(scale=0.1, size=(input_size, dof_size)).flatten()
        return LinearControllerGenotype(genotype, yaml_file)

    @staticmethod
    def develop_body(yaml_file: Optional[str]):
        if yaml_file is not None:
            assert os.path.exists(yaml_file)
            body = FixedBodyCreator(yaml_file).body
            actor, dof_ids = body.to_actor()
            return actor, dof_ids

        # fallback:
        # Hardcoded body; for now
        body = Body()
        body.core.front = ActiveHinge(math.pi / 2.0)
        body.core.front._absolute_rotation = 0
        body.core.front.attachment = ActiveHinge(math.pi / 2.0)
        body.core.front.attachment._absolute_rotation = 0
        body.core.left = ActiveHinge(math.pi / 2.0)
        body.core.left._absolute_rotation = 0
        body.core.left.attachment = ActiveHinge(math.pi / 2.0)
        body.core.left.attachment._absolute_rotation = 0
        body.core.right = ActiveHinge(math.pi / 2.0)
        body.core.right._absolute_rotation = 0
        body.core.right.attachment = ActiveHinge(math.pi / 2.0)
        body.core.right.attachment._absolute_rotation = 0
        body.core.back = ActiveHinge(math.pi / 2.0)
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
            DbGenotype(serialized_genome=np.array(o.genotype, dtype=float).tostring())
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
                np.fromstring(id_map[id].serialized_genome, dtype=float)
            )
            for id in ids
        ]
        return genotypes
