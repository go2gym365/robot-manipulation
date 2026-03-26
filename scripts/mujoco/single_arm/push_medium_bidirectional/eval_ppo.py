from stable_baselines3 import PPO

from envs.mujoco.single_arm.push_medium_bidirectional.env import (
    MujocoSingleArmPushMediumBidirectionalEnv,
)


def main():
    env = MujocoSingleArmPushMediumBidirectionalEnv()
    model = PPO.load(
        "models/mujoco/single_arm/push_medium_bidirectional/ppo_push_medium_bidirectional",
        device="cpu",
    )

    num_episodes = 10
    success_count = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0

        print(f"\n=== Eval Episode {ep + 1} ===")
        print(f"Initial obs: {obs}")
        print(f"Initial info: {info}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward

            print(
                f"step={step_count:3d} "
                f"action={action} "
                f"reward={reward:.3f} "
                f"obj_target_dist={info['object_target_distance']:.3f} "
                f"ee_obj_dist={info['ee_object_distance']:.3f} "
                f"success={info['is_success']}"
            )

            done = terminated or truncated

        if info["is_success"]:
            success_count += 1
        total_steps += step_count

        print(
            f"Episode done | total_reward={total_reward:.3f} | steps={step_count}"
        )

    print("\n=== Summary ===")
    print(f"Success rate: {success_count}/{num_episodes} = {success_count / num_episodes:.2f}")
    print(f"Average steps: {total_steps / num_episodes:.2f}")

    env.close()


if __name__ == "__main__":
    main()