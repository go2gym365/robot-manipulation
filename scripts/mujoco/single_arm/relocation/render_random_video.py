import os
import imageio.v2 as imageio

from envs.mujoco.single_arm.relocation.env import MujocoSingleArmRelocationEnv


def main():
    os.makedirs("videos/mujoco/single_arm/relocation", exist_ok=True)

    env = MujocoSingleArmRelocationEnv(render_mode="rgb_array")
    obs, info = env.reset()

    frames = []
    done = False
    step = 0

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    while not done and step < 200:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        print(
            f"step={step:3d} "
            f"reward={reward:.3f} "
            f"obj_target_dist={info['object_target_distance']:.3f} "
            f"ee_obj_dist={info['ee_object_distance']:.3f} "
            f"success={info['is_success']}"
        )

    output_path = "videos/mujoco/single_arm/relocation/random_rollout.mp4"
    imageio.mimsave(output_path, frames, fps=25)
    env.close()

    print(f"\nSaved video to: {output_path}")


if __name__ == "__main__":
    main()