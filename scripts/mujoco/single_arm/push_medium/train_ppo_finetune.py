from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.mujoco.single_arm.push_medium.env import MujocoSingleArmPushMediumEnv


def main():
    env = MujocoSingleArmPushMediumEnv()

    check_env(env, warn=True)

    pretrained_path = Path("models/mujoco/single_arm/push_easy/ppo_push_easy.zip")
    save_path = Path("models/mujoco/single_arm/push_medium/ppo_push_medium_ft")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if pretrained_path.exists():
        print(f"Loading pretrained model from: {pretrained_path}")
        model = PPO.load(str(pretrained_path), env=env, device="cpu")
        model.verbose = 1
    else:
        print("Pretrained model not found. Training from scratch instead.")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            n_steps=512,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            device="cpu",
        )

    model.learn(total_timesteps=150000)
    model.save(str(save_path))

    env.close()
    print(f"Saved fine-tuned model to: {save_path}.zip")


if __name__ == "__main__":
    main()