import os
import imageio.v2 as imageio
from stable_baselines3 import PPO

from envs.mujoco.single_arm.reach.env import MujocoSingleArmReachEnv


def main():
    os.makedirs("videos/mujoco/single_arm/reach", exist_ok=True)

    env = MujocoSingleArmReachEnv(render_mode="rgb_array")
    model = PPO.load("models/mujoco/single_arm/reach/ppo_arm_reach", device="cpu")

    obs, info = env.reset()
    done = False
    step = 0
    frames = []

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    while not done and step < 150:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        print(
            f"step={step:3d} "
            f"action={action} "
            f"reward={reward:.3f} "
            f"distance={info['distance']:.3f} "
            f"success={info['is_success']}"
        )

    output_path = "videos/mujoco/single_arm/reach/ppo_eval.mp4"
    imageio.mimsave(output_path, frames, fps=25)
    env.close()

    print(f"\nSaved video to: {output_path}")


if __name__ == "__main__":
    main()