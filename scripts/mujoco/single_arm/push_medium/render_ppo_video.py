from pathlib import Path

import imageio
from stable_baselines3 import PPO

from envs.mujoco.single_arm.push_easy.env import MujocoSingleArmPushEasyEnv


def main():
    model_path = "models/mujoco/single_arm/push_easy/ppo_push_easy"
    save_path = Path("videos/mujoco/single_arm/push_easy/ppo_eval.mp4")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    env = MujocoSingleArmPushEasyEnv(render_mode="rgb_array")
    model = PPO.load(model_path, device="cpu")

    obs, info = env.reset()
    frames = []

    total_reward = 0.0
    step_count = 0

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    print("\n=== Render Episode ===")
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