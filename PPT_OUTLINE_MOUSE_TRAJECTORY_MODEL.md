# Mouse Trajectory Generation Model PPT Outline

이 문서는 PPT 제작을 위한 발표 구성 초안입니다.  
동기와 문제의 중요성은 별도로 작성할 예정이므로 이 문서에서는 제외합니다.

## 1. 사용한 모델

### 모델 개요

사용한 기본 모델은 **Conditional Variational Autoencoder (CVAE) + GRU decoder** 구조입니다.

목표는 조건값을 입력받아 사람처럼 보이는 마우스 이동 궤적을 생성하는 것입니다.

입력 조건:

```text
A        이동 거리
W        목표 크기
ID       Fitts' law index of difficulty
duration 이동 시간
```

출력:

```text
normalized trajectory
(x_norm, y_norm) sequence
```

정규화 좌표계:

```text
start = (0, 0)
target = (1, 0)
```

즉 모델은 실제 화면 좌표를 직접 생성하는 것이 아니라, 시작점과 목표점을 기준으로 정규화된 궤적을 생성합니다.  
생성 후에는 원하는 화면 좌표계로 다시 변환할 수 있습니다.

### 모델 구조

```text
condition vector
        |
        v
CVAE latent variable z
        |
        v
GRU decoder
        |
        v
trajectory sequence
```

구성:

- Encoder: 실제 trajectory와 condition을 입력받아 latent distribution 추정
- Latent space: 같은 조건에서도 여러 가능한 사람 움직임을 표현
- Decoder: condition과 latent sample을 이용해 trajectory 생성
- Endpoint constraint: 시작점과 목표점을 고정하여 task 조건을 만족

## 2. 모델의 장점 및 방법론

### CVAE를 사용한 이유

마우스 궤적은 같은 조건에서도 하나의 정답만 존재하지 않습니다.

예:

```text
같은 A/W 조건에서도 사람마다, trial마다 다른 곡률과 미세 조정이 나타남
```

따라서 deterministic model보다 다양한 sample을 생성할 수 있는 generative model이 적합합니다.

CVAE의 장점:

- 같은 조건에서 여러 plausible trajectory 생성 가능
- 조건부 생성 가능
- latent variable로 사람 움직임의 다양성 표현
- 평균 궤적뿐 아니라 분산과 스타일 차이를 모델링 가능

### GRU decoder를 사용한 이유

마우스 궤적은 시간 순서를 가진 sequential data입니다.

GRU의 장점:

- 시간 순서가 있는 trajectory 생성에 적합
- 이전 위치 정보를 반영해 다음 위치 생성 가능
- 비교적 가볍고 학습 안정성이 좋음

### 핵심 방법론

초기에는 position reconstruction 중심으로 학습했습니다.

하지만 단순 위치 MSE는 다음 문제가 있었습니다.

```text
trajectory shape는 맞지만,
사람 손의 미세 조정과 jerk/acceleration dynamics가 사라짐
```

그래서 최종 모델에서는 위치뿐 아니라 다음 통계적 특성을 함께 맞추도록 손실함수를 확장했습니다.

최종 손실 구성:

```text
position reconstruction
KL divergence
acceleration statistic loss
jerk statistic loss
deviation statistic loss
path length statistic loss
peak velocity statistic loss
weak late acceleration statistic loss
weak late jerk statistic loss
```

핵심 아이디어:

```text
특정 timestamp의 acceleration을 그대로 복원하는 것이 아니라,
전체 trajectory 또는 batch 수준에서 실제 사람 움직임의 동역학 통계를 맞춘다.
```

## 3. 실험 구성

### 데이터 전처리

원본 CSV에는 trial 식별 정보가 부족했습니다.

문제:

- 여러 trajectory가 같은 trial key 아래로 합쳐짐
- 하나의 trial 안에서 `point_index`가 반복됨
- `ID`, `duration`, `t_norm` 컬럼이 없음

해결:

- `point_index` reset 지점을 기준으로 trajectory segment 분리
- `segment_index` 추가
- `ID`, `duration`, `t_norm` 재계산
- 모델 grouping key에 `segment_index` 추가

복구 결과:

```text
rows: 7,525,267
segments: 43,470
usable non-error trials: 41,431
```

### 평가 조건

초기에는 임의 조건으로 생성했지만, 이후 원본 데이터 조건을 따라가는 방식으로 변경했습니다.

원본 A/W 조건:

```text
A in {300, 301, 900, 901}
W in {20, 50, 120}
```

총 12개 조건에서 평가했습니다.

평가 방식:

```text
각 조건마다 temperature 0.0~0.9 생성
각 조건/temperature마다 20개 sample 생성
실제 데이터와 trajectory metric 비교
```

### 평가 지표

사용한 주요 지표:

```text
path_length_ratio
max_abs_deviation
peak_velocity
mean_acceleration
mean_jerk
```

해석:

- `path_length_ratio`: 실제보다 경로가 길거나 짧은지
- `max_abs_deviation`: 직선 경로에서 얼마나 벗어나는지
- `peak_velocity`: 최대 이동 속도
- `mean_acceleration`: 평균 가속도 크기
- `mean_jerk`: 가속도 변화량, 움직임의 미세 조정 정도

평가값은 대부분 다음 비율로 해석했습니다.

```text
generated / real
```

`1.0x`에 가까울수록 실제 데이터와 유사합니다.

## 4. 모델 변천사

### 4.1 Baseline

모델:

```text
CVAE + GRU
seq_len=64
smoothness loss 사용
```

문제:

- endpoint는 잘 맞춤
- 하지만 움직임이 너무 매끈함
- acceleration과 jerk가 실제보다 매우 낮음

해석:

```text
기본 CVAE+GRU는 평균적인 trajectory shape는 학습하지만,
사람 손의 미세 조정은 충분히 재현하지 못함
```

### 4.2 Dynamic-Loss Model

변경:

```text
seq_len: 64 -> 128
smoothness loss 제거
velocity reconstruction loss 추가
acceleration reconstruction loss 추가
```

기대효과:

- 시간 해상도 증가
- 속도/가속도 변화 복원
- over-smoothing 완화

결과:

```text
mean best gap: 0.421
path_length:   0.97x real
max_deviation: 0.79x real
peak_velocity: 1.22x real
acceleration:  0.24x real
jerk:          0.05x real
```

해석:

```text
pointwise velocity/acceleration loss는 오히려 미세 움직임을 평균화했다.
```

### 4.3 Statistic-Loss Model

변경:

```text
pointwise velocity/acceleration loss 제거
acceleration statistic loss 추가
jerk statistic loss 추가
deviation statistic loss 추가
```

기대효과:

- 특정 timestamp를 그대로 맞추지 않고 전체 동역학 분포를 맞춤
- 미세 움직임의 평균적인 크기와 분산을 실제와 유사하게 만듦

결과:

```text
mean best gap: 0.233
path_length:   1.01x real
max_deviation: 0.73x real
peak_velocity: 1.41x real
acceleration:  0.72x real
jerk:          0.68x real
```

해석:

```text
acceleration과 jerk가 크게 개선됨.
분포 기반 loss가 시간 dynamics를 살리는 데 효과적임.
```

### 4.4 Strong Late-Dynamics Model

변경:

```text
path_length_stat_loss 추가
late_acceleration_stat_loss 추가
late_jerk_stat_loss 추가
```

기대효과:

- 목표 근처에서 나타나는 미세 조정 강화
- jerk를 전체 궤적 크기가 아니라 target 근처 residual movement로 표현

결과:

```text
mean best gap: 0.416
path_length:   1.02x real
max_deviation: 0.97x real
peak_velocity: 2.24x real
acceleration:  0.88x real
jerk:          0.75x real
```

해석:

```text
acceleration과 deviation은 좋아졌지만,
late loss가 너무 강해서 목표 근처에서 급격히 꺾는 패턴이 생김.
peak velocity가 과도하게 증가함.
```

### 4.5 Weak Late + Peak-Velocity Model

변경:

```text
late loss weight를 1/3 이하로 감소
peak_velocity_stat_loss 추가
```

기대효과:

- late correction의 부작용 완화
- peak velocity 폭주 제어
- path/deviation/dynamics 간 균형 개선

결과:

```text
mean best gap: 0.156
path_length:   1.03x real
max_deviation: 0.97x real
peak_velocity: 1.18x real
acceleration:  0.79x real
jerk:          0.69x real
```

해석:

```text
현재까지 가장 좋은 균형.
trajectory shape, deviation, peak velocity가 실제와 가까워졌고,
acceleration/jerk도 baseline보다 크게 개선됨.
```

## 5. 결과 요약

모델별 실제 데이터 대비 결과:

| model | mean gap ↓ | path length | max deviation | peak velocity | acceleration | jerk |
|---|---:|---:|---:|---:|---:|---:|
| dynamic loss | 0.421 | 0.97x | 0.79x | 1.22x | 0.24x | 0.05x |
| stat loss | 0.233 | 1.01x | 0.73x | 1.41x | 0.72x | 0.68x |
| late loss | 0.416 | 1.02x | 0.97x | 2.24x | 0.88x | 0.75x |
| **late+peak loss** | **0.156** | **1.03x** | **0.97x** | **1.18x** | **0.79x** | **0.69x** |

핵심 결과:

- 최종 모델은 path length를 실제의 `1.03x`로 맞춤
- max deviation은 실제의 `0.97x`
- peak velocity는 실제의 `1.18x`
- acceleration은 실제의 `0.79x`
- jerk는 실제의 `0.69x`

즉 최종 모델은 trajectory shape와 speed profile을 실제와 가깝게 맞추면서, baseline에서 거의 사라졌던 dynamics를 상당 부분 복원했습니다.

## 6. 원본과의 유사점 및 평가

### 유사한 점

최종 모델은 다음 측면에서 원본 데이터와 유사합니다.

```text
1. 시작점과 목표점 구조를 유지
2. path length가 실제와 유사
3. 직선 경로에서 벗어나는 정도가 실제와 유사
4. peak velocity 폭주를 억제
5. 조건별 A/W 차이에 따라 궤적 형태가 달라짐
```

정량적으로 가장 잘 맞은 부분:

```text
path_length:   1.03x real
max_deviation: 0.97x real
peak_velocity: 1.18x real
```

### 아직 부족한 점

최종 모델에서도 acceleration과 jerk는 실제보다 낮습니다.

```text
acceleration: 0.79x real
jerk:         0.69x real
```

특히 다음 조건에서 acceleration/jerk가 낮게 나타났습니다.

```text
A=300/301, W=120
```

즉 큰 target width에서 실제 사람의 미세 조정을 완전히 재현하지는 못했습니다.

### 시각적 평가

대표 비교 이미지:

```text
generated/model_optimization_trajectory_evolution_with_real.png
```

이미지 구성:

```text
Real data
Baseline
Dynamic loss
Statistic loss
Strong late loss
Weak late + peak
```

해석:

- Baseline은 너무 평균적인 경향
- Statistic loss부터 궤적 다양성과 dynamics가 살아남
- Strong late loss는 목표 근처 급꺾임이 발생
- Weak late + peak 모델은 가장 안정적인 균형을 보임

## 7. 시사점

### 7.1 단순 position loss만으로는 부족하다

마우스 궤적은 위치 shape만 맞추면 되는 문제가 아닙니다.

사람처럼 보이려면 다음이 함께 맞아야 합니다.

```text
path shape
velocity profile
acceleration
jerk
target approach behavior
```

### 7.2 Pointwise dynamics loss는 한계가 있다

특정 timestamp의 velocity/acceleration을 직접 맞추는 방식은 실제 움직임의 다양성을 평균화할 수 있습니다.

이번 실험에서는 pointwise dynamic loss보다 statistic loss가 훨씬 효과적이었습니다.

### 7.3 분포 기반 loss가 중요하다

사람의 움직임은 하나의 deterministic trajectory가 아니라 분포입니다.

따라서 다음과 같은 통계 기반 평가/학습이 중요합니다.

```text
acceleration distribution
jerk distribution
path length distribution
deviation distribution
peak velocity distribution
```

### 7.4 목표 근처 correction은 조심해서 다뤄야 한다

목표 근처 미세 조정은 중요하지만, loss를 너무 강하게 주면 모델이 후반부에서 급격히 꺾어 목표를 맞추는 방식으로 학습할 수 있습니다.

따라서 late correction은 약하게 넣고, peak velocity나 path length 제어와 함께 사용해야 합니다.

## 8. 결론

이번 모델링 과정의 결론은 다음과 같습니다.

```text
CVAE+GRU는 조건부 마우스 궤적 생성에 적합한 기본 구조이다.
하지만 사람다운 trajectory를 만들기 위해서는 position reconstruction만으로는 부족하다.
acceleration, jerk, path length, deviation, peak velocity를 함께 고려하는 distribution-level loss가 필요하다.
```

최종 best 모델:

```text
models/mouse_trajectory_cvae_seq128_late_peak.pt
temperature=0.5
```

최종 모델은 실제 데이터 대비 다음 수준까지 도달했습니다.

```text
path_length:   1.03x real
max_deviation: 0.97x real
peak_velocity: 1.18x real
acceleration:  0.79x real
jerk:          0.69x real
```

즉 실제 궤적의 형태와 속도 특성은 상당히 잘 따라갔고, 사람 손의 미세 조정도 baseline보다 크게 개선했습니다.

## 9. 응용 가능성 및 기대효과

### 9.1 Human-like cursor simulation

사용자 실험이나 인터페이스 평가에서 사람과 유사한 cursor movement를 생성할 수 있습니다.

기대효과:

- 실제 사용자 데이터를 많이 수집하지 않고도 다양한 trajectory sample 생성
- UI target size, distance 조건별 interaction simulation 가능

### 9.2 HCI 모델링 및 평가

Fitts' law 기반 task에서 조건별 움직임 차이를 생성할 수 있습니다.

응용:

```text
target acquisition task simulation
pointing behavior analysis
cursor trajectory augmentation
```

### 9.3 Synthetic data generation

마우스 trajectory 데이터가 부족한 상황에서 synthetic trajectory를 생성할 수 있습니다.

기대효과:

- 모델 학습용 데이터 증강
- rare condition에 대한 trajectory 생성
- privacy-sensitive user movement data 대체 가능성

### 9.4 Adaptive UI / Accessibility

조건별 움직임 특성을 모델링하면 사용자의 motor behavior를 예측하거나 보정하는 데 활용할 수 있습니다.

가능한 방향:

- target size 조정
- cursor assistance
- motor-impaired user interaction simulation

## 10. 추가 개선 방향

### 조건별 temperature 선택

현재 전체 best temperature는 `0.5`이지만, 조건별 best temperature는 다릅니다.

개선:

```text
A/W/ID 조건에 따른 temperature lookup table
또는 condition-aware temperature predictor
```

### 조건별 loss weighting

일부 조건, 특히 `A=300/301, W=120`에서 acceleration/jerk가 낮습니다.

개선:

```text
condition-specific dynamics loss weighting
```

### Residual trajectory decoder

현재 모델은 전체 trajectory를 직접 생성합니다.

개선:

```text
trajectory = smooth base path + residual micro-adjustment
```

기대효과:

- 큰 궤적 loop 없이 jerk/acceleration 증가
- 목표 근처 미세 correction을 더 자연스럽게 표현

### Target approach 평가 강화

추가할 수 있는 평가 지표:

```text
late jerk
late acceleration
target entry angle
overshoot count
endpoint approach velocity
velocity profile correlation
```

이 지표들을 추가하면 “사람다운 움직임”을 더 설득력 있게 평가할 수 있습니다.

## 11. PPT 구성 제안

슬라이드 구성 예시:

```text
1. Problem setup
2. Model: CVAE + GRU
3. Why CVAE?
4. Data preprocessing and normalization
5. Evaluation metrics
6. Baseline result
7. Optimization history
8. Trajectory evolution image
9. Quantitative comparison table
10. Final model analysis
11. Similarities and remaining gaps
12. Implications
13. Applications and expected impact
14. Conclusion
```

주요 그림:

```text
generated/model_optimization_trajectory_evolution_with_real.png
generated/original_conditions_seq128_late_peak/evaluation_heatmap.png
generated/original_conditions_seq128_late_peak/trajectory_grid_temp0p5_points.png
```
