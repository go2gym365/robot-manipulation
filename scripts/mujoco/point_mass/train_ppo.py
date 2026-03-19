from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.mujoco.point_mass.env import MujocoReachEnv


def main():
    env = MujocoReachEnv()

    check_env(env, warn=True)

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

    model.learn(total_timesteps=100000)
    model.save("models/mujoco/ppo_reach")

    env.close()


if __name__ == "__main__":
    main()