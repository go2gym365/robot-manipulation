import numpy as np
from envs.mujoco.mujoco_reach_env import MujocoReachEnv

def main():
    env = MujocoReachEnv()

    for episode in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0.0  
        step = 0

        print(f"\n=== Heuristic Episode {episode + 1} ===")
        print("Initial obs:", obs)
        print("Initial info:", info)

        while not done:
            agent_pos = obs[:2]
            target_pos = obs[2:4]
            direction = target_pos = agent_pos

            action = np.clip(direction*3.0, -1.0, 1.0).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step += 1
            done = terminated or truncated

            print(
                f"step={step:3d} "
                f"action={action} "
                f"reward={reward:.3f} "
                f"distance={info['distance']:.3f} "
                f"success={info['is_success']}"
            )

        print(f"Episode done | total_reward={total_reward:.3f} | steps={step}")

    env.close()


if __name__ == "__main__":
    main()