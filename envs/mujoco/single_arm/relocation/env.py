import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


XML = """
<mujoco model="single_arm_relocation">
    <compiler angle="radian"/>
    <option timestep="0.02" gravity="0 0 0"/>

    <default>
        <geom friction="0.8 0.1 0.1" condim="3"/>
    </default>

    <worldbody>
        <!-- goal marker -->
        <body name="target" pos="0.75 0.0 0">
            <geom name="target_geom" type="sphere" size="0.05" rgba="0.9 0.2 0.2 0.5"/>
        </body>

        <!-- movable object -->
        <body name="object" pos="0.55 0.0 0">
            <joint name="obj_slide_x" type="slide" axis="1 0 0" damping="2.0"/>
            <joint name="obj_slide_y" type="slide" axis="0 1 0" damping="2.0"/>
            <geom name="object_geom" type="sphere" size="0.045" rgba="0.9 0.6 0.1 1"/>
        </body>

        <!-- arm -->
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="sphere" size="0.03" rgba="0.2 0.2 0.2 1"/>

            <body name="link1" pos="0 0 0">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1.0"/>
                <geom name="link1_geom" type="capsule" fromto="0 0 0 0.5 0 0" size="0.04" rgba="0.2 0.6 0.9 1"/>

                <body name="link2" pos="0.5 0 0">
                    <joint name="joint2" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1.0"/>
                    <geom name="link2_geom" type="capsule" fromto="0 0 0 0.35 0 0" size="0.035" rgba="0.2 0.8 0.4 1"/>

                    <!-- contact tip -->
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


class MujocoSingleArmRelocationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.success_threshold = 0.08

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

        # qpos: arm(2) + object(2), qvel: arm(2) + object(2)
        # obs = qpos_arm(2), qvel_arm(2), ee(2), object(2), target(2), ee->obj(2), obj->target(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.max_steps = 200
        self.step_count = 0

        self.prev_obj_target_dist = None
        self.prev_ee_obj_dist = None

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id][:2].copy()

    def _get_object_pos(self):
        return self.model.body_pos[self.object_body_id][:2].copy()

    def _get_target_pos(self):
        return self.model.body_pos[self.target_body_id][:2].copy()

    def _get_obs(self):
        arm_qpos = self.data.qpos[:2].copy()
        arm_qvel = self.data.qvel[:2].copy()
        ee_pos = self._get_ee_pos()
        obj_pos = self._get_object_pos()
        target_pos = self._get_target_pos()

        ee_to_obj = obj_pos - ee_pos
        obj_to_target = target_pos - obj_pos

        obs = np.concatenate(
            [arm_qpos, arm_qvel, ee_pos, obj_pos, target_pos, ee_to_obj, obj_to_target]
        ).astype(np.float32)
        return obs

    def _get_info(self):
        obj_pos = self._get_object_pos()
        target_pos = self._get_target_pos()
        obj_target_dist = np.linalg.norm(target_pos - obj_pos)

        ee_pos = self._get_ee_pos()
        ee_obj_dist = np.linalg.norm(obj_pos - ee_pos)

        return {
            "object_target_distance": float(obj_target_dist),
            "ee_object_distance": float(ee_obj_dist),
            "is_success": bool(obj_target_dist < self.success_threshold),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # arm init
        self.data.qpos[:2] = np.array([0.0, 0.0], dtype=np.float64)
        self.data.qvel[:2] = np.array([0.0, 0.0], dtype=np.float64)

        # object init
        obj_xy = self.np_random.uniform(
            low=np.array([0.45, -0.20]),
            high=np.array([0.65, 0.20]),
            size=(2,),
        )
        self.model.body_pos[self.object_body_id][:2] = obj_xy
        self.model.body_pos[self.object_body_id][2] = 0.0

        # target init
        target_xy = self.np_random.uniform(
            low=np.array([0.60, -0.35]),
            high=np.array([0.85, 0.35]),
            size=(2,),
        )
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

        # shaped reward:
        # 1) ee gets closer to object
        # 2) object gets closer to target
        reward = 0.3 * float(self.prev_ee_obj_dist - ee_obj_dist)
        reward += 1.0 * float(self.prev_obj_target_dist - obj_target_dist)
        reward -= 0.01

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