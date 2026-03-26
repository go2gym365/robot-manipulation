import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


XML = """
<mujoco model="single_arm_push_medium">
    <compiler angle="radian"/>
    <option timestep="0.02" gravity="0 0 0"/>

    <default>
        <geom friction="1.2 0.1 0.1" condim="3"/>
    </default>

    <worldbody>
        <body name="target" pos="0.72 0.0 0">
            <geom name="target_geom" type="sphere" size="0.05" rgba="0.9 0.2 0.2 0.5"/>
        </body>

        <body name="object" pos="0.56 0.0 0">
            <joint name="obj_slide_x" type="slide" axis="1 0 0" damping="2.5"/>
            <joint name="obj_slide_y" type="slide" axis="0 1 0" damping="2.5"/>
            <geom name="object_geom" type="sphere" size="0.045" rgba="0.9 0.6 0.1 1"/>
        </body>

        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="sphere" size="0.03" rgba="0.2 0.2 0.2 1"/>

            <body name="link1" pos="0 0 0">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1.2"/>
                <geom name="link1_geom" type="capsule" fromto="0 0 0 0.5 0 0" size="0.04" rgba="0.2 0.6 0.9 1"/>

                <body name="link2" pos="0.5 0 0">
                    <joint name="joint2" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1.2"/>
                    <geom name="link2_geom" type="capsule" fromto="0 0 0 0.35 0 0" size="0.035" rgba="0.2 0.8 0.4 1"/>

                    <body name="ee_body" pos="0.35 0 0">
                        <geom name="ee_geom" type="sphere" size="0.05" rgba="0.95 0.95 0.2 1"/>
                        <site name="ee_site" pos="0 0 0" size="0.02" rgba="1 1 0 1"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="motor1" joint="joint1" ctrlrange="-1 1" gear="30"/>
        <motor name="motor2" joint="joint2" ctrlrange="-1 1" gear="30"/>
    </actuator>
</mujoco>
"""


class MujocoSingleArmPushMediumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.success_threshold = 0.10

        self.contact_threshold_1 = 0.12
        self.contact_threshold_2 = 0.08
        self.contact_threshold_3 = 0.06

        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=480)

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        self.arm_qpos_slice = slice(0, 2)
        self.arm_qvel_slice = slice(0, 2)
        self.obj_qpos_slice = slice(2, 4)
        self.obj_qvel_slice = slice(2, 4)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.max_steps = 180
        self.step_count = 0

        self.prev_ee_obj_dist = None
        self.prev_obj_target_dist = None

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id][:2].copy()

    def _get_object_pos(self):
        return self.data.xpos[self.object_body_id][:2].copy()

    def _get_target_pos(self):
        return self.model.body_pos[self.target_body_id][:2].copy()

    def _get_obs(self):
        qpos = self.data.qpos[self.arm_qpos_slice].copy()
        qvel = self.data.qvel[self.arm_qvel_slice].copy()
        ee_pos = self._get_ee_pos()
        obj_pos = self._get_object_pos()
        target_pos = self._get_target_pos()

        ee_to_obj = obj_pos - ee_pos
        obj_to_target = target_pos - obj_pos

        obs = np.concatenate(
            [qpos, qvel, ee_pos, obj_pos, target_pos, ee_to_obj, obj_to_target]
        ).astype(np.float32)
        return obs

    def _get_info(self):
        ee_pos = self._get_ee_pos()
        obj_pos = self._get_object_pos()
        target_pos = self._get_target_pos()

        ee_obj_dist = np.linalg.norm(obj_pos - ee_pos)
        obj_target_dist = np.linalg.norm(target_pos - obj_pos)

        return {
            "ee_object_distance": float(ee_obj_dist),
            "object_target_distance": float(obj_target_dist),
            "is_success": bool(obj_target_dist < self.success_threshold),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self.arm_qpos_slice] = np.array([0.0, 0.0], dtype=np.float64)
        self.data.qvel[self.arm_qvel_slice] = np.array([0.0, 0.0], dtype=np.float64)

        # easy보다 y 범위를 넓힘
        obj_xy = self.np_random.uniform(
            low=np.array([0.53, -0.08]),
            high=np.array([0.61,  0.08]),
            size=(2,),
        )
        self.data.qpos[self.obj_qpos_slice] = obj_xy.astype(np.float64)
        self.data.qvel[self.obj_qvel_slice] = np.array([0.0, 0.0], dtype=np.float64)

        # target도 오른쪽이지만 y offset을 더 줌
        offset = self.np_random.uniform(
            low=np.array([0.11, -0.05]),
            high=np.array([0.17,  0.05]),
            size=(2,),
        )
        target_xy = obj_xy + offset
        target_xy[0] = np.clip(target_xy[0], 0.64, 0.84)
        target_xy[1] = np.clip(target_xy[1], -0.12, 0.12)

        self.model.body_pos[self.target_body_id][:2] = target_xy
        self.model.body_pos[self.target_body_id][2] = 0.0

        mujoco.mj_forward(self.model, self.data)

        ee_pos = self._get_ee_pos()
        obj_pos = self._get_object_pos()
        target_pos = self._get_target_pos()

        self.prev_ee_obj_dist = np.linalg.norm(obj_pos - ee_pos)
        self.prev_obj_target_dist = np.linalg.norm(target_pos - obj_pos)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action.astype(np.float64)

        mujoco.mj_step(self.model, self.data)

        ee_pos = self._get_ee_pos()
        obj_pos = self._get_object_pos()
        target_pos = self._get_target_pos()

        ee_obj_dist = np.linalg.norm(obj_pos - ee_pos)
        obj_target_dist = np.linalg.norm(target_pos - obj_pos)

        reward = 0.3 * float(self.prev_ee_obj_dist - ee_obj_dist)
        reward += 3.0 * float(self.prev_obj_target_dist - obj_target_dist)
        reward -= 0.01

        entered_contact_1 = (
            self.prev_ee_obj_dist >= self.contact_threshold_1
            and ee_obj_dist < self.contact_threshold_1
        )
        entered_contact_2 = (
            self.prev_ee_obj_dist >= self.contact_threshold_2
            and ee_obj_dist < self.contact_threshold_2
        )
        entered_contact_3 = (
            self.prev_ee_obj_dist >= self.contact_threshold_3
            and ee_obj_dist < self.contact_threshold_3
        )

        if entered_contact_1:
            reward += 0.03
        if entered_contact_2:
            reward += 0.07
        if entered_contact_3:
            reward += 0.12

        entered_target_14 = (
            self.prev_obj_target_dist >= 0.14 and obj_target_dist < 0.14
        )
        entered_target_12 = (
            self.prev_obj_target_dist >= 0.12 and obj_target_dist < 0.12
        )

        if entered_target_14:
            reward += 0.05
        if entered_target_12:
            reward += 0.08

        terminated = bool(obj_target_dist < self.success_threshold)
        truncated = bool(self.step_count >= self.max_steps)

        if terminated:
            reward += 10.0

        self.prev_ee_obj_dist = ee_obj_dist
        self.prev_obj_target_dist = obj_target_dist

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None