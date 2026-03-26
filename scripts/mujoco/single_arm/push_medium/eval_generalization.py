import numpy as np
from stable_baselines3 import PPO

from envs.mujoco.single_arm.push_medium.env import MujocoSingleArmPushMediumEnv


class MujocoSingleArmPushMediumGeneralizationEvalEnv(MujocoSingleArmPushMediumEnv):
    """
    Train env는 그대로 두고,
    eval에서만 target/object distribution을 조금 더 넓혀서
    generalization을 확인하는 환경.
    """

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        # 원래 reset에서 세팅한 상태를 다시 덮어쓴다.
        # mj_resetData 후 직접 새 distribution으로 샘플링
        import mujoco

        mujoco.mj_resetData(self.model, self.data)

        # arm init
        self.data.qpos[:2] = np.array([0.0, 0.0], dtype=np.float64)
        self.data.qvel[:2] = np.array([0.0, 0.0], dtype=np.float64)

        # object는 medium보다 약간 더 다양한 위치
        obj_xy = self.np_random.uniform(
            low=np.array([0.50, -0.14]),
            high=np.array([0.64, 0.14]),
            size=(2,),
        )
        self.data.qpos[2:4] = obj_xy.astype(np.float64)
        self.data.qvel[2:4] = np.array([0.0, 0.0], dtype=np.float64)

        # target은 "주로 오른쪽"이지만 y variation을 더 크게 줌
        offset = self.np_random.uniform(
            low=np.array([0.10, -0.16]),
            high=np.array([0.22, 0.16]),
            size=(2,),
        )
        target_xy = obj_xy + offset
        target_xy[0] = np.clip(target_xy[0], 0.58, 0.85)
        target_xy[1] = np.clip(target_xy[1], -0.25, 0.25)

        self.model.body_pos[self.target_body_id][:2] = target_xy
        self.model.body_pos[self.target_body_id][2] = 0.0

        mujoco.mj_forward(self.model, self.data)

        ee_pos = self._get_ee_pos()
        obj_pos = self._get_object_pos()
        target_pos = self._get_target_pos()

        self.prev_ee_obj_dist = np.linalg.norm(obj_pos - ee_pos)
        self.prev_obj_target_dist = np.linalg.norm(target_pos - obj_pos)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info


def main():
    env = MujocoSingleArmPushMediumGeneralizationEvalEnv()
    model = PPO.load(
        "models/mujoco/single_arm/push_medium/ppo_push_medium_ft",
        device="cpu",
    )

    num_episodes = 10
    success_count = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0

        print(f"\n=== Generalization Eval Episode {ep + 1} ===")
        print(f"Initial obs: {obs}")
        print(f"Initial info: {info}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

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

    print("\n=== Generalization Summary ===")
    print(f"Success rate: {success_count}/{num_episodes} = {success_count / num_episodes:.2f}")
    print(f"Average steps: {total_steps / num_episodes:.2f}")

    env.close()


if __name__ == "__main__":
    main()