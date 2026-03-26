from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO

from envs.mujoco.single_arm.push_medium.env import MujocoSingleArmPushMediumEnv


class MujocoSingleArmPushMediumBidirectionalEvalEnv(MujocoSingleArmPushMediumEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        import mujoco

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # arm init
        self.data.qpos[:2] = np.array([0.0, 0.0], dtype=np.float64)
        self.data.qvel[:2] = np.array([0.0, 0.0], dtype=np.float64)

        # object 위치
        obj_xy = self.np_random.uniform(
            low=np.array([0.50, -0.14]),
            high=np.array([0.64, 0.14]),
            size=(2,),
        )
        self.data.qpos[2:4] = obj_xy.astype(np.float64)
        self.data.qvel[2:4] = np.array([0.0, 0.0], dtype=np.float64)

        # 50% right / 50% left
        if self.np_random.random() < 0.5:
            offset = self.np_random.uniform(
                low=np.array([0.10, -0.16]),
                high=np.array([0.22, 0.16]),
                size=(2,),
            )
            direction = "right"
        else:
            offset = self.np_random.uniform(
                low=np.array([-0.22, -0.16]),
                high=np.array([-0.10, 0.16]),
                size=(2,),
            )
            direction = "left"

        target_xy = obj_xy + offset
        target_xy[0] = np.clip(target_xy[0], 0.35, 0.85)
        target_xy[1] = np.clip(target_xy[1], -0.25, 0.25)

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
        info["target_side"] = direction
        return obs, info


def main():
    model_path = "models/mujoco/single_arm/push_medium/ppo_push_medium_ft"
    save_path = Path("videos/mujoco/single_arm/push_medium/ppo_eval_bidirectional.mp4")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    env = MujocoSingleArmPushMediumBidirectionalEvalEnv(render_mode="rgb_array")
    model = PPO.load(model_path, device="cpu")

    obs, info = env.reset()
    frames = []

    total_reward = 0.0
    step_count = 0

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    print("\n=== Render Episode (Bidirectional Generalization) ===")
    print(f"Initial obs: {obs}")
    print(f"Initial info: {info}")

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        print(
            f"step={step_count:3d} "
            f"action={action} "
            f"reward={reward:.3f} "
            f"obj_target_dist={info['object_target_distance']:.3f} "
            f"ee_obj_dist={info['ee_object_distance']:.3f} "
            f"success={info['is_success']}"
        )

        done = terminated or truncated

    imageio.mimsave(save_path, frames, fps=50)
    env.close()

    print(f"\nEpisode done | total_reward={total_reward:.3f} | steps={step_count}")
    print(f"Saved video to: {save_path}")


if __name__ == "__main__":
    main()