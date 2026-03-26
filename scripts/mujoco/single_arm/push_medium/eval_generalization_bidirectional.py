import numpy as np
from stable_baselines3 import PPO

from envs.mujoco.single_arm.push_medium.env import MujocoSingleArmPushMediumEnv


class MujocoSingleArmPushMediumBidirectionalEvalEnv(MujocoSingleArmPushMediumEnv):
    """
    더 강한 generalization 평가용 env.
    target이 object의 오른쪽뿐 아니라 왼쪽에도 올 수 있게 만든다.
    학습은 기존 medium FT 모델 그대로 사용한다.
    """

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        import mujoco

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # arm init
        self.data.qpos[:2] = np.array([0.0, 0.0], dtype=np.float64)
        self.data.qvel[:2] = np.array([0.0, 0.0], dtype=np.float64)

        # object 위치: 기존보다 약간 넓은 범위
        obj_xy = self.np_random.uniform(
            low=np.array([0.50, -0.14]),
            high=np.array([0.64, 0.14]),
            size=(2,),
        )
        self.data.qpos[2:4] = obj_xy.astype(np.float64)
        self.data.qvel[2:4] = np.array([0.0, 0.0], dtype=np.float64)

        # target 방향:
        # 50%는 오른쪽, 50%는 왼쪽
        if self.np_random.random() < 0.5:
            # right side
            offset = self.np_random.uniform(
                low=np.array([0.10, -0.16]),
                high=np.array([0.22, 0.16]),
                size=(2,),
            )
        else:
            # left side
            offset = self.np_random.uniform(
                low=np.array([-0.22, -0.16]),
                high=np.array([-0.10, 0.16]),
                size=(2,),
            )

        target_xy = obj_xy + offset
        target_xy[0] = np.clip(target_xy[0], 0.35, 0.85)
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
    env = MujocoSingleArmPushMediumBidirectionalEvalEnv()
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

        print(f"\n=== Bidirectional Generalization Eval Episode {ep + 1} ===")
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

    print("\n=== Bidirectional Generalization Summary ===")
    print(f"Success rate: {success_count}/{num_episodes} = {success_count / num_episodes:.2f}")
    print(f"Average steps: {total_steps / num_episodes:.2f}")

    env.close()


if __name__ == "__main__":
    main()