from stable_baselines3 import PPO

from envs.mujoco.single_arm.push_easy.env import MujocoSingleArmPushEasyEnv


def main():
    env = MujocoSingleArmPushEasyEnv()
    model = PPO.load("models/mujoco/single_arm/push_easy/ppo_push_easy", device="cpu")

    num_episodes = 10
    success_count = 0
    total_steps = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        final_success = False

        print(f"\n=== Eval Episode {episode + 1} ===")
        print("Initial obs:", obs)
        print("Initial info:", info)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step += 1
            done = terminated or truncated
            final_success = info["is_success"]

            print(
                f"step={step:3d} "
                f"action={action} "
                f"reward={reward:.3f} "
                f"obj_target_dist={info['object_target_distance']:.3f} "
                f"ee_obj_dist={info['ee_object_distance']:.3f} "
                f"success={info['is_success']}"
            )

        if final_success:
            success_count += 1
        total_steps += step

        print(f"Episode done | total_reward={total_reward:.3f} | steps={step}")

    print("\n=== Summary ===")
    print(f"Success rate: {success_count}/{num_episodes} = {success_count / num_episodes:.2f}")
    print(f"Average steps: {total_steps / num_episodes:.2f}")

    env.close()


if __name__ == "__main__":
    main()