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