from envs.toy.reach_env import ReachEnv

def main():
    env = ReachEnv()

    for episode in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"\n=== Episode {episode + 1} ===")
        print("Initial obs:", obs)
        print("Initial info:", info)

        while not done:
            action = env.action_space.sample()
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


if __name__ == "__main__":
    main()