### 26.03.10 (wed)
- random policy는 max_steps까지는 도달하지만 성공한 상태는 아님
- heuristic policy는 현재 위치와 목표 위치 차이를 계산해서 target direction으로 이동하기 때문에 빠르게 성공
- step size를 0.05에서 0.2로 바꾸자 훨씬 빨리 종료되었음
