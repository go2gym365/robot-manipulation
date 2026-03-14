from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.toy.reach_env import ReachEnv

def main():
    env =  ReachEnv()

    check_env(env, warn=True)

    model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99
        )
    model.learn(total_timesteps=20000)
    model.save("models/toy/ppo_reach")


if __name__ == "__main__":
    main()