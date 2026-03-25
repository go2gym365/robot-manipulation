import os
import imageio.v2 as imageio
from stable_baselines3 import PPO

from envs.mujoco.single_arm.push_easy.env import MujocoSingleArmPushEasyEnv


def main():
    os.makedirs("videos/mujoco/single_arm/push_easy", exist_ok=True)

    env = MujocoSingleArmPushEasyEnv(render_mode="rgb_array")
    model = PPO.load("models/mujoco/single_arm/push_easy/ppo_push_easy", device="cpu")

    obs, info = env.reset()
    done = False
    step = 0
    frames = []

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    while not done and step < 180:
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
            f"obj_target_dist={info['object_target_distance']:.3f} "
            f"ee_obj_dist={info['ee_object_distance']:.3f} "
            f"success={info['is_success']}"
        )

    output_path = "videos/mujoco/single_arm/push_easy/ppo_eval.mp4"
    imageio.mimsave(output_path, frames, fps=25)
    env.close()

    print(f"\nSaved video to: {output_path}")


if __name__ == "__main__":
    main()