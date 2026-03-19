import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


XML = """
<mujoco model="single_arm_reach">
    <compiler angle="radian"/>
    <option timestep="0.02" gravity="0 0 0"/>

    <worldbody>
        <body name="target" pos="0.8 0.0 0">
            <geom name="target_geom" type="sphere" size="0.04" rgba="0.9 0.2 0.2 1"/>
        </body>

        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="sphere" size="0.03" rgba="0.2 0.2 0.2 1"/>

            <body name="link1" pos="0 0 0">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1.0"/>
                <geom name="link1_geom" type="capsule" fromto="0 0 0 0.5 0 0" size="0.04" rgba="0.2 0.6 0.9 1"/>

                <body name="link2" pos="0.5 0 0">
                    <joint name="joint2" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1.0"/>
                    <geom name="link2_geom" type="capsule" fromto="0 0 0 0.4 0 0" size="0.035" rgba="0.2 0.8 0.4 1"/>
                    <site name="ee_site" pos="0.4 0 0" size="0.03" rgba="1 1 0 1"/>
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


class MujocoSingleArmReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=480)

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        # obs = [qpos(2), qvel(2), ee_pos(2), target_pos(2), rel(2)] = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # action = 2 joint motor controls
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.max_steps = 150
        self.step_count = 0

    def _get_ee_pos(self):
        # site_xpos shape: (nsite, 3)
        return self.data.site_xpos[self.ee_site_id][:2].copy()

    def _get_target_pos(self):
        return self.model.body_pos[self.target_body_id][:2].copy()

    def _get_obs(self):
        qpos = self.data.qpos[:2].copy()
        qvel = self.data.qvel[:2].copy()
        ee_pos = self._get_ee_pos()
        target_pos = self._get_target_pos()
        rel = target_pos - ee_pos

        obs = np.concatenate([qpos, qvel, ee_pos, target_pos, rel]).astype(np.float32)
        return obs

    def _get_info(self):
        ee_pos = self._get_ee_pos()
        target_pos = self._get_target_pos()
        dist = np.linalg.norm(target_pos - ee_pos)
        return {
            "distance": float(dist),
            "is_success": bool(dist < 0.06),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # initial joint state
        self.data.qpos[:] = np.array([0.0, 0.0], dtype=np.float64)
        self.data.qvel[:] = np.array([0.0, 0.0], dtype=np.float64)

        # reachable target region in XY plane
        target_xy = self.np_random.uniform(
            low=np.array([0.2, -0.6]),
            high=np.array([0.85, 0.6]),
            size=(2,)
        )
        self.model.body_pos[self.target_body_id][:2] = target_xy
        self.model.body_pos[self.target_body_id][2] = 0.0

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action.astype(np.float64)

        mujoco.mj_step(self.model, self.data)

        ee_pos = self._get_ee_pos()
        target_pos = self._get_target_pos()
        dist = np.linalg.norm(target_pos - ee_pos)

        reward = float(-dist)
        terminated = bool(dist < 0.06)
        truncated = bool(self.step_count >= self.max_steps)

        if terminated:
            reward += 10.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        self.renderer.update_scene(self.data)
        frame = self.renderer.render()
        return frame

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None