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

## Single Arm Push Easy v2
- push task의 난이도를 낮추기 위해 object와 target 배치를 거의 일직선(오른쪽 방향)으로 제한한 curriculum 버전을 실험했다.
- object의 y 범위를 줄이고, target을 object의 오른쪽에만 생성하도록 설정하여 task를 사실상 “한 방향 push” 문제로 단순화했다.
- 그 결과 PPO evaluation에서 success rate 1.00 (10/10), average steps 14.30을 기록하였다.
- 이를 통해 PPO가 pushing behavior 자체를 학습할 수 없었던 것이 아니라, 기존 setting에서는 exploration과 task geometry가 너무 어려웠다는 점을 확인했다.
- 즉 manipulation에서는 reward shaping뿐 아니라 task curriculum 설계가 매우 중요하다는 점을 실험적으로 확인했다.


이번 push_easy_v2 실험이 성공한 주요 원인은 두 가지이다.
첫째, sparse reward 대신 end-effector의 접근, contact 형성, object의 target 방향 progress를 반영하는 dense reward를 설계하여 PPO가 중간 단계의 행동도 학습할 수 있게 했다.
둘째, object와 target의 배치를 거의 일직선으로 제한하여 task geometry를 단순화함으로써 exploration 난이도를 크게 낮췄다.
즉 이번 결과는 reward shaping과 curriculum 설계가 함께 작동했을 때 manipulation task에서도 PPO가 안정적으로 수렴할 수 있음을 보여준다.

## Push Medium Fine-tuning Result
- push_medium task를 scratch로 학습했을 때는 success rate가 매우 낮고 policy가 반복적인 saturated action에 빠지는 문제가 있었다.
- 이를 해결하기 위해 push_easy에서 학습한 policy를 초기값으로 사용하여 push_medium에 fine-tuning을 적용했다.
- fine-tuning 결과 evaluation에서 success rate 1.00 (10/10), average steps 16.60을 기록하였다.
- 이 결과는 pushing skill이 easy task에서 medium task로 전이될 수 있으며, manipulation 학습에서 curriculum learning이 매우 효과적임을 보여준다.
- 즉 PPO가 medium task를 전혀 학습하지 못하는 것이 아니라, 적절한 skill initialization과 단계적 난이도 증가가 필요하다는 점을 확인했다.



- 학습된 push_medium fine-tuned policy의 일반화 성능을 확인하기 위해, 평가 시 target y variation과 object 위치 분포를 더 넓힌 generalization evaluation을 추가했다.
- 이 실험의 목적은 현재 policy가 특정 target 방향에 과적합된 것인지, 아니면 일정 수준의 일반적인 pushing skill을 학습했는지 확인하는 것이다.
- train distribution은 유지하고 eval distribution만 확장함으로써, curriculum으로 학습한 skill의 전이 가능성을 분석한다.