from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.mujoco.single_arm.push_medium_bidirectional.env import (
    MujocoSingleArmPushMediumBidirectionalEnv,
)


def main():
    env = MujocoSingleArmPushMediumBidirectionalEnv()
    check_env(env, warn=True)

    model = PPO.load(
        "models/mujoco/single_arm/push_medium/ppo_push_medium_ft",
        env=env,
        device="cpu",
    )

    model.learn(total_timesteps=100000)
    model.save("models/mujoco/single_arm/push_medium_bidirectional/ppo_push_medium_bidirectional")

    env.close()


if __name__ == "__main__":
    main()