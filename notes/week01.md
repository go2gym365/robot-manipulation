## Progress
- Week 1: Implemented a toy 2D ReachEnv, tested random and heuristic rollouts, and verified MuJoCo state stepping with a smoke test.

### 1
- random policy는 max_steps까지는 도달하지만 성공한 상태는 아님
- heuristic policy는 현재 위치와 목표 위치 차이를 계산해서 target direction으로 이동하기 때문에 빠르게 성공
- step size를 0.05에서 0.2로 바꾸자 훨씬 빨리 종료되었음

### 2
- PPO 학습 결과를 확인했을 때, target 근처까지는 접근하지만 success가 계속 false로 남는 문제가 있었다
로그를 확인해보니 에이전트가 목표점 근처까지 갔다가 정확히 멈추지 못하고 주변을 맴도는 패턴이 나타났다
원인은 MuJoCo 환경에서 damping이 없어 속도가 충분히 줄지 않는 동적 특성 때문이라고 판단함
이를 해결하기 위해 slide joint에 damping을 추가했고, 이후 정책이 목표점 근처에서 더 안정적으로 수렴하여 success가 정상적으로 발생하며 성공률 10/10을 확임했다

### 3
구현 내용:
- 2개의 회전 관절을 가진 간단한 single arm Mujoco 모델을 생성
- observation에는 joint position, joint velocity, end-effector position, target position, relative vector를 포함했다.
- action은 두 관절에 대한 actuator control input으로 정의했다.
- reward는 end-effector와 target 사이 거리 기반으로 설계했고, success 시 bonus를 주도록 했다.

결과:
기본 distance reward만 사용한 single arm reach 실험에서는 success rate는 0.8 수준이었지만, 평균 success episode 길이가 약 60 step으로 다소 길었다. 
이후 progress reward(이전 step 대비 target과의 거리 감소량)와 작은 step penalty를 추가하여 실험을 다시 진행했다.  
그 결과 success rate는 동일하게 0.8 수준을 유지했지만, 평균 episode 길이는 약 46.8 step으로 감소했다.  
즉, progress reward는 성공률 자체를 크게 높이지는 않았지만, policy가 target에 더 빠르게 도달하도록 유도하는 데는 도움이 되는 것으로 보였다.