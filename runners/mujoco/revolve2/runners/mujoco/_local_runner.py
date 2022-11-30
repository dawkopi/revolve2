import math
import tempfile
from typing import List, Set

import cv2
import mujoco_viewer
import numpy as np

import mujoco

try:
    import logging

    old_len = len(logging.root.handlers)

    from dm_control import mjcf

    new_len = len(logging.root.handlers)

    assert (
        old_len + 1 == new_len
    ), "dm_control not adding logging handler as expected. Maybe they fixed their annoying behaviour? https://github.com/deepmind/dm_control/issues/314https://github.com/deepmind/dm_control/issues/314"

    logging.root.removeHandler(logging.root.handlers[-1])
except Exception as e:
    print("Failed to fix absl logging bug", e)
    pass

from pyrr import Quaternion, Vector3
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    BatchResults,
    Environment,
    EnvironmentResults,
    EnvironmentState,
    Runner,
)
from typing import Callable, Optional


class LocalRunner(Runner):
    """Runner for simulating using Mujoco."""

    _headless: bool

    def __init__(self, headless: bool = False):
        """
        Initialize this object.

        :param headless: If True, the simulation will not be rendered. This drastically improves performance.
        """
        self._headless = headless

    def run_batch_sync(
        self, batch: Batch, is_healthy: Optional[Callable] = None, video_path: str = ""
    ) -> BatchResults:
        return self._run_batch(batch, is_healthy=is_healthy, video_path=video_path)

    async def run_batch(
        self, batch: Batch, is_healthy: Optional[Callable] = None, video_path: str = ""
    ) -> BatchResults:
        """
        Run the provided batch by simulating each contained environment.

        :param batch: The batch to run.
        :param is_healthy: function that evaluates whether the robot is in a "healthy state". (If not the simulation should be terminated).
        :returns: List of simulation states in ascending order of time.
        """
        return self._run_batch(batch, is_healthy=is_healthy, video_path=video_path)

    def _run_batch(
        self, batch: Batch, is_healthy: Optional[Callable] = None, video_path: str = ""
    ) -> BatchResults:
        logging.info("Starting simulation batch with mujoco.")
        video_fps = 24
        control_step = 1 / batch.control_frequency * 2
        sample_step = 1 / batch.sampling_frequency
        video_step = 1 / video_fps

        results = BatchResults([EnvironmentResults([]) for _ in batch.environments])

        for env_index, env_descr in enumerate(batch.environments):
            xml_string = self._make_mjcf(env_descr)
            model = mujoco.MjModel.from_xml_string(xml_string)

            # TODO initial dof state
            data = mujoco.MjData(model)

            initial_targets = [
                dof_state
                for posed_actor in env_descr.actors
                for dof_state in posed_actor.dof_states
            ]
            self._set_dof_targets(data, initial_targets)

            for posed_actor in env_descr.actors:
                posed_actor.dof_states

            if not self._headless or video_path:
                viewer = mujoco_viewer.MujocoViewer(
                    model,
                    data,
                )
                if video_path:
                    # viewer._render_every_frame = False  # save a lot of time
                    # http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vid = cv2.VideoWriter(
                        video_path,
                        fourcc,
                        video_fps,
                        (viewer.viewport.width, viewer.viewport.height),
                    )

            last_control_time = 0.0
            last_sample_time = 0.0
            last_video_time = 0.0  # time at which last video frame was saved

            # sample initial state
            results.environment_results[env_index].environment_states.append(
                EnvironmentState(
                    0.0, [], [], self._get_actor_states(env_descr, data, model)
                )
            )

            actions = []
            action_diffs = []
            last_action = None
            while (time := data.time) < batch.simulation_time:
                # do control if it is time
                if time >= last_control_time + control_step:
                    last_control_time = math.floor(time / control_step) * control_step
                    control = ActorControl()

                    # get actor state so we can read joint angles/velocities
                    actor_state = self._get_actor_states(
                        env_descr,
                        data,
                        model,
                        ground_contacts=False,
                    )[0]

                    logging.debug(f"actor height = {actor_state.position.z:0.3f}")
                    # TODO: the exact time at which we terminate isn't tracked exactly
                    #   (whatever last sample time was is taken to be the duration)
                    if is_healthy is not None:
                        if not is_healthy(actor_state):
                            # end the simulation
                            logging.info(
                                f"stopping sim at time {time:0.3f} due to unhealthy actor!"
                            )
                            break

                    batch.control(
                        env_index,
                        actor_state,
                        control_step,
                        control,
                    )
                    actor_targets = control._dof_targets
                    action = control._dof_targets[0][1]
                    actions.append(action)
                    if last_action is None:
                        action_diffs.append(action)
                    else:
                        action_diffs.append(
                            [action[i] - last_action[i] for i in range(len(action))]
                        )
                    last_action = action
                    actor_targets.sort(key=lambda t: t[0])
                    targets = [
                        target
                        for actor_target in actor_targets
                        for target in actor_target[1]
                    ]
                    # set target angles of the joints
                    self._set_dof_targets(data, targets)

                # sample state if it is time
                if time >= last_sample_time + sample_step:
                    last_sample_time = int(time / sample_step) * sample_step
                    env_state = EnvironmentState(
                        time,
                        actions,
                        action_diffs,
                        self._get_actor_states(
                            env_descr,
                            data,
                            model,
                        ),
                    )
                    results.environment_results[env_index].environment_states.append(
                        env_state
                    )
                    actions = []
                    action_diffs = []

                # step simulation
                mujoco.mj_step(model, data)

                if not self._headless:
                    viewer.render()

                # capture video frame if it's time
                if video_path and time >= last_video_time + video_step:
                    last_video_time = int(time / video_step) * video_step

                    if self._headless:
                        # ensure render is called anyways
                        viewer._hide_menu = (
                            viewer._hide_menu or video_path
                        )  # hack (don't show overlay in video)
                        viewer.render()

                    # https://github.com/deepmind/mujoco/issues/285 (see also record.cc)
                    img = np.empty(
                        (viewer.viewport.height, viewer.viewport.width, 3),
                        dtype=np.uint8,
                    )

                    mujoco.mjr_readPixels(
                        rgb=img,
                        depth=None,
                        viewport=viewer.viewport,
                        con=viewer.ctx,
                    )
                    img = np.flip(img, axis=0)  # img is upside down initially
                    vid.write(img)
                    # matplotlib.image.imsave("/tmp/first.png", img)

            if not self._headless or video_path:
                viewer.close()
            if video_path:
                vid.release()

            # sample one final time
            results.environment_results[env_index].environment_states.append(
                EnvironmentState(
                    time,
                    actions,
                    action_diffs,
                    self._get_actor_states(env_descr, data, model),
                )
            )

        return results

    @staticmethod
    def _make_mjcf(env_descr: Environment) -> str:
        env_mjcf = mjcf.RootElement(model="environment")

        env_mjcf.compiler.angle = "radian"

        env_mjcf.option.timestep = 0.0005
        env_mjcf.option.integrator = "RK4"

        env_mjcf.option.gravity = [0, 0, -9.81]

        env_mjcf.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=[10, 10, 1],
            rgba=[0.2, 0.2, 0.2, 1],
        )
        env_mjcf.worldbody.add(
            "light",
            pos=[0, 0, 100],
            ambient=[1.0, 1.0, 1.0],
            directional=True,
            castshadow=False,
        )
        env_mjcf.visual.headlight.active = 0

        for actor_index, posed_actor in enumerate(env_descr.actors):
            urdf = physbot_to_urdf(
                posed_actor.actor,
                f"robot_{actor_index}",
                Vector3(),
                Quaternion(),
            )

            model = mujoco.MjModel.from_xml_string(urdf)

            # mujoco can only save to a file, not directly to string,
            # so we create a temporary file.
            botfile = tempfile.NamedTemporaryFile(
                mode="r+", delete=False, suffix=".urdf"
            )
            mujoco.mj_saveLastXML(botfile.name, model)
            robot = mjcf.from_file(botfile)
            botfile.close()

            force_range = 0.9
            for joint in posed_actor.actor.joints:
                # robot.actuator.add(
                #     "intvelocity",
                #     actrange="-1.57 1.57",
                #     kp=5.0,
                #     forcerange=f"{-force_range} {force_range}",
                #     joint=robot.find(namespace="joint", identifier=joint.name),
                # )
                robot.find(namespace="joint", identifier=joint.name).armature = "0.01"
                robot.find(namespace="joint", identifier=joint.name).damping = "0.01"
                robot.actuator.add(
                    "position",
                    kp=1.0,
                    # kp=5.0,
                    # kp=0.2,
                    ctrlrange="-1.0 1.0",
                    forcerange=f"{-force_range} {force_range}",
                    joint=robot.find(
                        namespace="joint",
                        identifier=joint.name,
                    ),
                )
                # robot.actuator.add(
                #     "velocity",
                #     kv=0.001,
                #     # kv=0.4,
                #     # kv=0.05,
                #     ctrlrange="-1.0 1.0",
                #     forcerange=f"{-force_range} {force_range}",
                #     joint=robot.find(namespace="joint", identifier=joint.name),
                # )

            attachment_frame = env_mjcf.attach(robot)
            attachment_frame.add("freejoint")
            attachment_frame.pos = [
                posed_actor.position.x,
                posed_actor.position.y,
                posed_actor.position.z,
            ]

            attachment_frame.quat = [
                posed_actor.orientation.x,
                posed_actor.orientation.y,
                posed_actor.orientation.z,
                posed_actor.orientation.w,
            ]

        xml = env_mjcf.to_xml_string()
        if not isinstance(xml, str):
            raise RuntimeError("Error generating mjcf xml.")

        return xml

    @classmethod
    def _get_actor_states(
        cls,
        env_descr: Environment,
        data: mujoco.MjData,
        model: mujoco.MjModel,
        ground_contacts: bool = True,
    ) -> List[ActorState]:
        return [
            cls._get_actor_state(i, data, model, ground_contacts)
            for i in range(len(env_descr.actors))
        ]

    @staticmethod
    def _get_actor_state(
        robot_index: int,
        data: mujoco.MjData,
        model: mujoco.MjModel,
        ground_contacts: bool = True,  # whether to track ground contacts (could be slow)
    ) -> ActorState:
        robotname = f"robot_{robot_index}/"  # the slash is added by dm_control. ugly but deal with it
        bodyid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            robotname,
        )
        assert bodyid >= 0

        qindex = model.body_jntadr[bodyid]

        # explicitly copy because the Vector3 and Quaternion classes don't copy the underlying structure
        position = Vector3([n for n in data.qpos[qindex : qindex + 3]])
        orientation = Quaternion([n for n in data.qpos[qindex + 3 : qindex + 3 + 4]])

        geomids: Optional[Set[int]] = None
        numgeoms: Optional[int] = None
        if ground_contacts:
            contacts = data.contact
            # https://mujoco.readthedocs.io/en/latest/overview.html#floating-objects
            groundid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

            geomids = set()  # ids of geometries in contact with ground
            for c in contacts:
                if groundid in [c.geom1, c.geom2] and c.geom1 != c.geom2:
                    otherid = c.geom1 if c.geom2 == groundid else c.geom2
                    othername = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, otherid
                    )
                    # logging.debug(f"found ground collision with geom ID: {otherid} (name '{othername}')")
                    if not othername.startswith(robotname):
                        continue  # ensure contact is with part of the robot (e.g. not obstacle and ground)
                    geomids.add(otherid)

            # if len(geomids) > 0:
            #    logging.debug(f"found {len(geomids)} total geoms in contact with ground")
            #    names = list([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, curid) for curid in geomids])
            #    logging.debug(names)

        # track states of hinge joints
        #   for reference see mjmodel.h and simulate.cc:makejoint()
        jnt_hinge_indices = [
            i
            for i, val in enumerate(model.jnt_type)
            if val == mujoco.mjtJoint.mjJNT_HINGE
        ]
        hinge_angles = [data.qpos[model.jnt_qposadr[i]] for i in jnt_hinge_indices]
        hinge_vels = [data.qvel[model.jnt_dofadr[i]] for i in jnt_hinge_indices]

        return ActorState(
            position,
            orientation,
            geomids,
            model.ngeom,
            hinge_angles,
            hinge_vels,
        )

    @staticmethod
    def _set_dof_targets(data: mujoco.MjData, targets: List[float]) -> None:
        if len(targets) == len(data.ctrl):
            for i, target in enumerate(targets):
                data.ctrl[i] = target
        elif len(targets) * 2 != len(data.ctrl):
            raise RuntimeError(
                "Number of target dofs doesn't match the number of actuators"
            )
        else:
            for i, target in enumerate(targets):
                data.ctrl[2 * i] = target
                data.ctrl[2 * i + 1] = 0
