import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


XML = """
<mujoco model="point_mass_reach">
    <option timestep="0.02" gravity="0 0 0"/>

    <worldbody>
        <body name="point" pos="0 0 0">
            <joint name="slide_x" type="slide" axis="1 0 0" damping="1.0"/>
            <joint name="slide_y" type="slide" axis="0 1 0" damping="1.0"/>
            <geom name="agent" type="sphere" size="0.04" rgba="0.2 0.6 0.9 1"/>
        </body>

        <body name="target" pos="0.3 0.3 0">
            <geom name="target_geom" type="sphere" size="0.03" rgba="0.9 0.2 0.2 1"/>
        </body>
    </worldbody>

    <actuator>
        <motor joint="slide_x" ctrlrange="-1 1" gear="1"/>
        <motor joint="slide_y" ctrlrange="-1 1" gear="1"/>
    </actuator>
</mujoco>
"""


class MujocoReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=480)

        # observation = [agent_x, agent_y, target_x, target_y, rel_x, rel_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # action = 2D control input
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.max_steps = 100
        self.step_count = 0

        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

    def _get_agent_pos(self):
        return self.data.qpos[:2].copy()

    def _get_target_pos(self):
        return self.model.body_pos[self.target_body_id][:2].copy()

    def _get_obs(self):
        agent_pos = self.data.qpos[:2].copy()
        agent_vel = self.data.qvel[:2].copy()
        target_pos = self._get_target_pos()
        rel = target_pos - agent_pos
        obs = np.concatenate([agent_pos, agent_vel, target_pos, rel]).astype(np.float32)
        return obs

    def _get_info(self):
        agent_pos = self._get_agent_pos()
        target_pos = self._get_target_pos()
        dist = np.linalg.norm(target_pos - agent_pos)
        return {
            "distance": float(dist),
            "is_success": bool(dist < 0.08),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # 초기 agent 위치
        self.data.qpos[:] = np.array([0.0, 0.0], dtype=np.float64)
        self.data.qvel[:] = np.array([0.0, 0.0], dtype=np.float64)

        # target 랜덤 위치
        target_xy = self.np_random.uniform(low=-0.5, high=0.5, size=(2,))
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

        agent_pos = self._get_agent_pos()
        target_pos = self._get_target_pos()
        dist = np.linalg.norm(target_pos - agent_pos)

        reward = float(-dist)
        terminated = bool(dist < 0.08)
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