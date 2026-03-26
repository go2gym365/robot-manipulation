- single arm relocation task를 시도했지만, PPO는 object를 target으로 이동시키는 정책을 학습하지 못했다.
- evaluation 로그를 보면 end-effector는 움직였지만 object-target distance는 거의 변하지 않았고, success rate는 0/10이었다.
- 이를 통해 reaching과 object manipulation 사이에는 난이도 차이가 크며, relocation 전에 object touch와 같은 더 쉬운 contact task가 필요하다고 판단했다.


- PPO 학습 로그에서 success rate가 0.9~0.97 수준까지 상승했고, 평균 episode 길이도 약 18~20 step 수준까지 감소했다.  
- 이는 single arm touch task에서 policy가 object에 빠르고 안정적으로 접근하는 행동을 학습했음을 의미한다.  
- 또한 explained variance도 대체로 높은 값을 유지하여, value estimation 역시 비교적 안정적으로 이루어졌음을 확인했다.

- single arm push_easy task를 PPO로 학습시켰지만, object-target distance가 거의 줄어들지 않았고 success rate는 0/10으로 나타났다.
- end-effector는 object 근처까지 접근했지만, object를 목표 방향으로 실제로 미는 정책은 학습되지 않았다.
- 이를 통해 touch와 pushing 사이에도 난이도 차이가 크며, pushing에서는 contact 이후의 방향성 있는 manipulation이 추가로 필요하다는 점을 확인했다.
- 다음 단계에서는 heuristic pushing controller를 통해 환경 자체에서 object relocation이 가능한지 먼저 검증할 예정이다.


- push_easy 환경의 object state tracking과 reward 구조를 수정한 뒤, PPO evaluation에서 success rate는 0/10에서 1/10으로 소폭 개선되었다.
- end-effector는 object 쪽으로 접근하는 행동을 일부 학습했지만, object-target distance를 안정적으로 줄이는 pushing policy는 아직 형성되지 않았다.
- 즉 현재 단계에서는 접근(skill of reaching object)은 부분적으로 학습되었지만, contact 이후 target 방향으로 물체를 밀어내는 manipulation은 여전히 어렵다는 점을 확인했다.
- 다음 단계에서는 push task를 더 쉬운 curriculum으로 바꾸기 위해 object와 target 배치를 거의 일직선(오른쪽 방향)으로 제한하는 실험을 진행할 예정이다.