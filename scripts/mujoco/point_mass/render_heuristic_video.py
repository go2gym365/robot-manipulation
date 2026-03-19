import os
import imageio.v2 as imageio
import numpy as np

from envs.mujoco.point_mass.env import MujocoReachEnv


def main():
    os.makedirs("videos/mujoco", exist_ok=True)

    env = MujocoReachEnv(render_mode="rgb_array")
    obs, info = env.reset()

    frames = []
    done = False
    step = 0

    # 첫 프레임
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    while not done and step < 100:
        agent_pos = obs[:2]
        target_pos = obs[2:4]
        direction = target_pos - agent_pos
        action = np.clip(direction * 3.0, -1.0, 1.0).astype(np.float32)

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

    output_path = "videos/mujoco/heuristic_reach.mp4"
    imageio.mimsave(output_path, frames, fps=25)
    env.close()

    print(f"\nSaved video to: {output_path}")


if __name__ == "__main__":
    main()