# Mouse Trajectory Model Optimization Log

이 문서는 마우스 궤적 생성 모델을 최적화해 온 과정을 발표/PPT 제작용으로 정리한 기록입니다.

각 단계는 다음 질문에 답하도록 구성했습니다.

- 어떤 모델 조합을 사용했는가?
- 이전 버전과 무엇이 달라졌는가?
- 왜 바꾸었는가?
- 기대한 효과는 무엇인가?
- 실제 평가 결과는 어땠는가?
- 다음 개선 방향은 무엇인가?

## 0. 현재 결론

현재까지 가장 좋은 모델은 다음입니다.

```text
model: models/mouse_trajectory_cvae_seq128_late_peak.pt
recommended temperature: 0.4~0.6
best overall temperature: 0.5
```

현재 best 모델은 **CVAE + GRU** 구조를 유지하면서, 손실함수에 다음 요소를 조합한 버전입니다.

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

기존 실제 데이터 대비 최종 성능은 다음과 같습니다.

| model | mean gap ↓ | path length | max deviation | peak velocity | acceleration | jerk |
|---|---:|---:|---:|---:|---:|---:|
| dynamic loss | 0.421 | 0.97x | 0.79x | 1.22x | 0.24x | 0.05x |
| stat loss | 0.233 | 1.01x | 0.73x | 1.41x | 0.72x | 0.68x |
| late loss | 0.416 | 1.02x | 0.97x | 2.24x | 0.88x | 0.75x |
| **late+peak loss** | **0.156** | **1.03x** | **0.97x** | **1.18x** | **0.79x** | **0.69x** |

해석:

- path length와 max deviation은 실제 데이터에 매우 가까워졌다.
- peak velocity는 late loss 실험에서 크게 폭주했지만, peak velocity statistic loss를 추가한 뒤 `1.18x`로 안정화됐다.
- acceleration과 jerk는 baseline 대비 크게 개선됐지만, 아직 실제보다 약간 낮다.
- 전체 균형 기준으로는 `late+peak loss` 모델이 가장 좋다.

## 1. 데이터 복구 단계

### 문제

입력 파일:

```text
normalized_trajectory_points.csv
```

처음 CSV에는 학습에 필요한 trial 식별 정보가 부족했습니다.

기존 모델은 다음 key로 trajectory를 묶었습니다.

```python
GROUP_KEYS = ("subject", "subfolder", "condition_index", "trial_number")
```

하지만 실제 CSV에서는 여러 원본 trajectory가 같은 key 아래로 합쳐져 있었습니다. 그 결과 하나의 trial 안에서 `point_index=0,1,2...`가 여러 번 반복되는 문제가 있었습니다.

또한 CSV에 다음 컬럼이 없었습니다.

```text
ID
duration
t_norm
```

이 상태로 학습하면 모델 조건값 일부가 기본값으로 들어가고, 서로 다른 궤적들이 하나의 trial로 섞이는 문제가 생깁니다.

### 수정

스크립트:

```text
repair_normalized_csv.py
```

수정 방식:

- `point_index`가 다시 0 또는 이전 값 이하로 돌아가는 지점을 새 trajectory segment로 분리
- `segment_index` 추가
- `A`, `W`로부터 `ID = log2(A / W + 1)` 재계산
- segment 길이로부터 `duration`, `t_norm` 재계산
- 모델의 `GROUP_KEYS`에 `segment_index` 추가

수정 후:

```python
GROUP_KEYS = ("subject", "subfolder", "condition_index", "trial_number", "segment_index")
```

### 결과

```text
output: normalized_trajectory_points_fixed.csv
rows: 7,525,267
segments: 43,470
usable non-error trials: 41,431
```

이후 모든 학습과 평가는 `normalized_trajectory_points_fixed.csv`를 기준으로 수행했습니다.

## 2. Baseline 모델

### 모델 조합

기본 구조:

```text
CVAE + GRU
```

입력 조건:

```text
A
W
ID
duration
```

출력:

```text
normalized trajectory points: (x_norm, y_norm)
```

구조:

- Encoder: GRU
- Latent variable: CVAE latent vector `z`
- Decoder: GRU
- 시작점은 `(0, 0)`으로 강제
- 끝점은 `(1, 0)`으로 강제

### 학습 설정

```text
model: models/mouse_trajectory_cvae.pt
seq_len: 64
beta: 0.001
smooth_weight: 0.02
epochs: 200
```

최종 로그:

```text
epoch 0200 train=0.000864 val=0.000878
```

### 평가

처음에는 다음 조건으로 생성했습니다.

```text
A=500
W=40
ID=3.75
duration=650
```

하지만 이 조건은 원본 데이터에 존재하는 A/W 조합이 아니었습니다.

원본 데이터 조건은 다음과 같았습니다.

```text
A in {300, 301, 900, 901}
W in {20, 50, 120}
```

따라서 이후 평가는 원본 A/W 조건을 따라가도록 수정했습니다.

### 결과 해석

baseline은 endpoint를 잘 맞추지만, 다음 문제가 있었습니다.

- 외삽 조건에서는 trajectory가 너무 크게 휘거나 길어짐
- 원본 조건 안에서도 움직임이 너무 매끈함
- acceleration과 jerk가 실제 데이터보다 크게 낮음

대표 in-distribution 평가:

```text
condition: A=301, W=20

real      path=1.168  dev=0.081  end=0.016  jerk=0.0062
temp=0.6 path=1.104  dev=0.056  end=0.000  jerk=0.0007
temp=0.8 path=1.066  dev=0.092  end=0.000  jerk=0.0007
```

핵심 문제:

```text
모양은 만들지만 사람 손의 미세 조정과 시간 dynamics가 거의 사라짐
```

## 3. Dynamic-Loss 모델

### 이전 버전 대비 변경점

Baseline에서 다음을 변경했습니다.

```text
seq_len: 64 -> 128
smooth_weight: 0.02 -> 0
beta: 0.001 -> 0.0001
velocity reconstruction loss 추가
acceleration reconstruction loss 추가
```

### 왜 바꾸었는가?

Baseline은 너무 매끈한 궤적을 만들었습니다.  
이는 다음 두 가지 때문이라고 판단했습니다.

- `seq_len=64`가 실제 움직임의 미세 변화를 충분히 보존하지 못함
- smoothness loss가 acceleration/jerk를 더 작게 만들 가능성이 큼

따라서 시간 해상도를 늘리고, velocity/acceleration을 직접 복원하도록 했습니다.

### 기대효과

- 더 세밀한 시간 변화 보존
- 속도 변화와 가속도 변화가 실제에 가까워질 것
- jerk collapse 완화

### 학습 설정

```powershell
python mouse_trajectory_model.py train `
  --csv normalized_trajectory_points_fixed.csv `
  --output models/mouse_trajectory_cvae_seq128_dyn.pt `
  --seq-len 128 `
  --smooth-weight 0 `
  --beta 0.0001 `
  --velocity-weight 1.0 `
  --acceleration-weight 0.5 `
  --epochs 200
```

최종 로그:

```text
epoch 0200 train=0.000245 val=0.000296
```

### 평가 결과

원본 12개 A/W 조건에서 평가했습니다.

```text
A in {300, 301, 900, 901}
W in {20, 50, 120}
temperature: 0.0~0.9
samples per condition/temp: 20
```

결과:

| metric | generated / real |
|---|---:|
| mean best gap | 0.421 |
| path length | 0.97x |
| max deviation | 0.79x |
| peak velocity | 1.22x |
| acceleration | 0.24x |
| jerk | 0.05x |

### 해석

좋아진 점:

- path length는 실제와 가까움
- peak velocity도 크게 나쁘지 않음

나쁜 점:

- acceleration은 실제의 `0.24x`
- jerk는 실제의 `0.05x`
- 즉 시간 dynamics가 거의 살아나지 않음

결론:

```text
pointwise velocity/acceleration reconstruction은 미세 움직임을 살리기보다 평균화하는 경향이 있음
```

## 4. Statistic-Loss 모델

### 이전 버전 대비 변경점

Dynamic-loss 모델에서 pointwise velocity/acceleration loss를 제거하고, 통계 기반 loss를 추가했습니다.

제거:

```text
velocity reconstruction loss
acceleration reconstruction loss
```

추가:

```text
acceleration_stat_loss
jerk_stat_loss
deviation_stat_loss
```

### 왜 바꾸었는가?

마우스 움직임의 미세 조정은 특정 timestamp에서 정확히 같은 acceleration을 복원하는 문제가 아닙니다.  
중요한 것은 궤적 전체 또는 batch 단위에서 acceleration/jerk의 **분포와 크기**가 실제와 비슷한지입니다.

따라서 pointwise reconstruction 대신 다음 통계를 맞추도록 했습니다.

- acceleration magnitude의 평균/분산
- jerk magnitude의 평균/분산
- lateral deviation의 평균/분산

### 기대효과

- 궤적의 전체 모양은 유지
- 특정 real trajectory를 복사하지 않음
- 생성 궤적의 미세 움직임 양이 실제와 가까워짐
- over-smoothing 완화

### 학습 설정

```powershell
python mouse_trajectory_model.py train `
  --csv normalized_trajectory_points_fixed.csv `
  --output models/mouse_trajectory_cvae_seq128_stat.pt `
  --seq-len 128 `
  --smooth-weight 0 `
  --beta 0.0001 `
  --velocity-weight 0 `
  --acceleration-weight 0 `
  --acceleration-stat-weight 0.001 `
  --jerk-stat-weight 0.001 `
  --deviation-stat-weight 0.0005 `
  --epochs 120
```

최종 로그:

```text
epoch 0120 train=0.000266 val=0.000254
```

### 평가 결과

Temperature ranking:

```text
temp=0.7 mean_gap=0.271
temp=0.9 mean_gap=0.278
temp=0.8 mean_gap=0.281
```

Best-per-condition aggregate:

| metric | generated / real |
|---|---:|
| mean best gap | 0.233 |
| path length | 1.01x |
| max deviation | 0.73x |
| peak velocity | 1.41x |
| acceleration | 0.72x |
| jerk | 0.68x |

### 실제로 잘 바뀌었는가?

Dynamic-loss 모델 대비:

| metric | dynamic | stat |
|---|---:|---:|
| mean best gap | 0.421 | 0.233 |
| acceleration | 0.24x | 0.72x |
| jerk | 0.05x | 0.68x |

해석:

- acceleration/jerk collapse가 크게 완화됨
- path length는 실제와 거의 동일
- 하지만 max deviation은 아직 실제보다 낮고, peak velocity는 실제보다 높음

결론:

```text
통계 기반 loss는 시간 dynamics를 살리는 데 효과적이었다.
```

## 5. Strong Late-Dynamics 모델

### 이전 버전 대비 변경점

Statistic-loss 모델에 다음을 추가했습니다.

```text
path_length_stat_loss
late_acceleration_stat_loss
late_jerk_stat_loss
```

### 왜 바꾸었는가?

실제 사람 마우스 움직임에서는 목표 근처에서 미세 조정이 많이 발생할 가능성이 높습니다.  
따라서 jerk를 전체 trajectory 크기로 키우는 대신, 마지막 구간에서 residual correction처럼 만들고 싶었습니다.

### 기대효과

- path length는 실제 분포에 묶어둠
- 목표 근처의 acceleration/jerk 증가
- 전체 loop 없이 후반 미세 조정 강화

### 학습 설정

```powershell
python mouse_trajectory_model.py train `
  --csv normalized_trajectory_points_fixed.csv `
  --output models/mouse_trajectory_cvae_seq128_late.pt `
  --seq-len 128 `
  --smooth-weight 0 `
  --beta 0.0001 `
  --velocity-weight 0 `
  --acceleration-weight 0 `
  --acceleration-stat-weight 0.002 `
  --jerk-stat-weight 0.003 `
  --deviation-stat-weight 0.001 `
  --path-length-stat-weight 0.001 `
  --late-acceleration-stat-weight 0.001 `
  --late-jerk-stat-weight 0.002 `
  --epochs 120
```

최종 로그:

```text
epoch 0120 train=0.001243 val=0.000967
```

### 평가 결과

Temperature ranking:

```text
temp=0.0 mean_gap=0.421
temp=0.1 mean_gap=0.431
temp=0.2 mean_gap=0.458
```

Best-per-condition aggregate:

| metric | generated / real |
|---|---:|
| mean best gap | 0.416 |
| path length | 1.02x |
| max deviation | 0.97x |
| peak velocity | 2.24x |
| acceleration | 0.88x |
| jerk | 0.75x |

### 실제로 잘 바뀌었는가?

좋아진 점:

- max deviation이 `0.97x`로 실제와 매우 가까워짐
- acceleration이 `0.88x`까지 개선됨
- jerk도 `0.75x`로 약간 개선됨

나빠진 점:

- peak velocity가 `2.24x`로 폭주
- high temperature에서 궤적이 빠르게 망가짐
- best temperature가 `0.0~0.1`로 내려가 다양성이 거의 사라짐
- 시각적으로 후반부에 급하게 꺾어 목표를 맞추는 패턴이 나타남

결론:

```text
late loss는 방향성은 일부 맞았지만 너무 강하게 넣으면 후반 급꺾임과 peak velocity 폭주를 만든다.
```

## 6. Weak Late-Dynamics + Peak-Velocity 모델

### 이전 버전 대비 변경점

Strong late-dynamics 모델에서 다음을 바꾸었습니다.

```text
late_acceleration_stat_weight: 0.001 -> 0.0003
late_jerk_stat_weight: 0.002 -> 0.0007
peak_velocity_stat_loss 추가
```

즉, late loss는 1/3 이하로 낮추고, peak velocity를 직접 제어했습니다.

### 왜 바꾸었는가?

Strong late 모델은 acceleration은 좋아졌지만, 목표 근처에서 급격히 보정하는 방식으로 학습되었습니다.  
그 결과 peak velocity가 실제보다 너무 커졌습니다.

따라서 다음 전략으로 바꾸었습니다.

- late loss는 약하게 유지
- path length와 deviation은 계속 제어
- peak velocity statistic을 실제와 맞춰 속도 폭주 방지

### 기대효과

- late model의 peak velocity 폭주 완화
- stat model보다 path/deviation 균형 개선
- temperature가 다시 `0.4~0.6` 정도의 실사용 가능한 범위로 돌아옴
- 다양성을 유지하면서 실제 데이터에 더 가까운 평균 지표 달성

### 학습 설정

```powershell
python mouse_trajectory_model.py train `
  --csv normalized_trajectory_points_fixed.csv `
  --output models/mouse_trajectory_cvae_seq128_late_peak.pt `
  --seq-len 128 `
  --smooth-weight 0 `
  --beta 0.0001 `
  --velocity-weight 0 `
  --acceleration-weight 0 `
  --acceleration-stat-weight 0.0015 `
  --jerk-stat-weight 0.002 `
  --deviation-stat-weight 0.0008 `
  --path-length-stat-weight 0.001 `
  --peak-velocity-stat-weight 0.001 `
  --late-acceleration-stat-weight 0.0003 `
  --late-jerk-stat-weight 0.0007 `
  --epochs 120
```

최종 로그:

```text
epoch 0120 train=0.000631 val=0.000436
```

### 평가 결과

Temperature ranking:

```text
temp=0.5 mean_gap=0.166
temp=0.4 mean_gap=0.178
temp=0.6 mean_gap=0.187
temp=0.3 mean_gap=0.208
temp=0.7 mean_gap=0.236
```

Best-per-condition aggregate:

| metric | generated / real |
|---|---:|
| mean best gap | 0.156 |
| path length | 1.03x |
| max deviation | 0.97x |
| peak velocity | 1.18x |
| acceleration | 0.79x |
| jerk | 0.69x |

### 실제로 잘 바뀌었는가?

이전 모델들과 비교:

| model | mean gap ↓ | path length | max deviation | peak velocity | acceleration | jerk |
|---|---:|---:|---:|---:|---:|---:|
| dynamic loss | 0.421 | 0.97x | 0.79x | 1.22x | 0.24x | 0.05x |
| stat loss | 0.233 | 1.01x | 0.73x | 1.41x | 0.72x | 0.68x |
| late loss | 0.416 | 1.02x | 0.97x | 2.24x | 0.88x | 0.75x |
| **late+peak loss** | **0.156** | **1.03x** | **0.97x** | **1.18x** | **0.79x** | **0.69x** |

해석:

- 현재까지 가장 좋은 정량 결과
- peak velocity 폭주를 크게 완화
- path length와 max deviation이 실제와 매우 가까움
- acceleration/jerk는 strong late 모델보다 조금 낮지만, 전체 균형은 훨씬 좋음
- temperature `0.5`가 가장 안정적

결론:

```text
현재 best 모델은 late+peak loss 모델이다.
```

## 7. 현재 best 모델의 조건별 평가

모델:

```text
models/mouse_trajectory_cvae_seq128_late_peak.pt
```

조건별 best temperature와 실제 대비 비율:

| cond | A | W | best temp | gap ↓ | path | dev | peak | acc | jerk |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 300 | 20 | 0.6 | 0.118 | 1.02x | 0.98x | 1.09x | 0.80x | 0.74x |
| 01 | 300 | 50 | 0.6 | 0.160 | 1.04x | 1.18x | 1.00x | 0.74x | 0.69x |
| 02 | 300 | 120 | 0.4 | 0.267 | 1.06x | 0.74x | 0.97x | 0.54x | 0.48x |
| 03 | 301 | 20 | 0.6 | 0.131 | 1.05x | 0.98x | 1.20x | 0.84x | 0.78x |
| 04 | 301 | 50 | 0.5 | 0.144 | 1.05x | 1.04x | 1.05x | 0.74x | 0.68x |
| 05 | 301 | 120 | 0.8 | 0.257 | 1.15x | 1.00x | 1.20x | 0.58x | 0.49x |
| 06 | 900 | 20 | 0.5 | 0.156 | 1.00x | 0.79x | 1.41x | 1.10x | 0.94x |
| 07 | 900 | 50 | 0.4 | 0.094 | 0.97x | 0.92x | 1.16x | 0.97x | 0.82x |
| 08 | 900 | 120 | 0.5 | 0.161 | 1.02x | 1.08x | 1.29x | 0.89x | 0.69x |
| 09 | 901 | 20 | 0.6 | 0.130 | 1.01x | 1.00x | 1.45x | 1.12x | 0.93x |
| 10 | 901 | 50 | 0.5 | 0.112 | 1.01x | 0.98x | 1.34x | 1.04x | 0.85x |
| 11 | 901 | 120 | 0.4 | 0.139 | 0.99x | 1.03x | 1.14x | 0.83x | 0.65x |

조건별 해석:

- `A=900/901` 조건에서는 대체로 실제와 잘 맞는다.
- `A=300/301, W=120` 조건에서는 acceleration/jerk가 아직 낮다.
- 큰 target width에서는 실제 움직임이 더 단순하거나 조건별 분산이 다르게 나타날 수 있어, 조건별 loss weighting이 필요할 수 있다.

## 8. 주요 산출물

학습 모델:

```text
models/mouse_trajectory_cvae.pt
models/mouse_trajectory_cvae_seq128_dyn.pt
models/mouse_trajectory_cvae_seq128_stat.pt
models/mouse_trajectory_cvae_seq128_late.pt
models/mouse_trajectory_cvae_seq128_late_peak.pt
```

평가 스크립트:

```text
evaluate_trajectories.py
evaluate_original_conditions.py
```

복구 스크립트:

```text
repair_normalized_csv.py
```

주요 평가 결과:

```text
generated/original_conditions_seq128_dyn/evaluation_by_condition.csv
generated/original_conditions_seq128_dyn/evaluation_heatmap.png
generated/original_conditions_seq128_stat/evaluation_by_condition.csv
generated/original_conditions_seq128_stat/evaluation_heatmap.png
generated/original_conditions_seq128_late/evaluation_by_condition.csv
generated/original_conditions_seq128_late/evaluation_heatmap.png
generated/original_conditions_seq128_late_peak/evaluation_by_condition.csv
generated/original_conditions_seq128_late_peak/evaluation_heatmap.png
```

대표 시각화:

```text
generated/original_conditions_seq128_stat/trajectory_grid_temp0p7_points.png
generated/original_conditions_seq128_stat/trajectory_grid_temp0p8_points.png
generated/original_conditions_seq128_stat/trajectory_grid_temp0p9_points.png
generated/original_conditions_seq128_late/trajectory_grid_temp0p0_points.png
generated/original_conditions_seq128_late_peak/trajectory_grid_temp0p4_points.png
generated/original_conditions_seq128_late_peak/trajectory_grid_temp0p5_points.png
generated/original_conditions_seq128_late_peak/trajectory_grid_temp0p6_points.png
```

## 9. 개선방안

현재 best 모델은 전체 균형이 가장 좋지만, 아직 개선 여지가 있습니다.

### 9.1 조건별 temperature 설정

현재 전체적으로는 `temperature=0.5`가 가장 좋지만, 조건별 best temperature는 다릅니다.

예:

```text
A=300, W=20 -> temp=0.6
A=300, W=120 -> temp=0.4
A=301, W=120 -> temp=0.8
A=900, W=50 -> temp=0.4
```

개선안:

- A/W/ID 조건에 따라 temperature를 자동 선택
- 또는 condition별 temperature lookup table 사용

기대효과:

- 전체 평균 gap 추가 감소
- 조건별 부자연스러운 variation 완화

### 9.2 조건별 loss weighting

문제:

- `A=300/301, W=120` 조건에서 acceleration/jerk가 낮다.
- 반면 `A=900/901` 조건은 상대적으로 잘 맞는다.

개선안:

- 조건별로 acceleration/jerk loss weight를 다르게 적용
- 특히 short distance + large target 조건에 별도 보정

기대효과:

- 약한 조건의 dynamics 개선
- 전체 모델이 큰 거리 조건에만 맞춰지는 문제 완화

### 9.3 Residual decoder 구조

현재 decoder는 전체 trajectory 좌표를 직접 생성합니다.

개선안:

```text
trajectory = base_path + residual_micro_adjustment
```

- base path: 부드러운 큰 궤적
- residual: 작은 미세 조정, jerk/acceleration 담당

기대효과:

- 전체 궤적 loop를 만들지 않고도 jerk/acceleration 증가
- 사람 손의 미세 correction을 더 자연스럽게 표현

### 9.4 Late correction의 위치 제어

Strong late 모델은 목표 근처 correction을 너무 강하게 학습해 후반 급꺾임을 만들었습니다.

개선안:

- late window를 고정 `last 30%`가 아니라 `last 10~25%`로 실험
- correction 크기에 upper bound penalty 추가
- target 근처에서만 residual이 발생하도록 spatial weighting 적용

기대효과:

- 목표 근처 micro-adjustment는 살리고, 큰 후반 꺾임은 방지

### 9.5 더 세밀한 평가 지표

현재 지표:

```text
path_length_ratio
max_abs_deviation
peak_velocity
mean_acceleration
mean_jerk
```

추가하면 좋은 지표:

```text
late jerk only
late acceleration only
overshoot count
target-entry behavior
endpoint approach angle
velocity profile correlation
```

기대효과:

- “사람답다”는 주장을 더 구체적으로 검증 가능
- 단순 평균 jerk가 아니라 목표 근처 행동을 별도로 설명 가능

## 10. PPT용 핵심 메시지

짧게 요약하면 다음 흐름입니다.

```text
1. 데이터가 trial 단위로 섞여 있어서 segment_index 기반으로 복구했다.
2. baseline CVAE+GRU는 endpoint는 맞췄지만 너무 매끈했다.
3. pointwise velocity/acceleration loss는 미세 움직임을 살리지 못했다.
4. acceleration/jerk statistic loss가 dynamics를 크게 개선했다.
5. late dynamics를 강하게 넣으면 acceleration은 좋아지지만 peak velocity가 폭주했다.
6. late loss를 약하게 줄이고 peak velocity를 제어하자 현재까지 가장 좋은 균형을 얻었다.
```

최종 주장:

```text
The final model best matched real trajectories when combining CVAE+GRU with distribution-level trajectory dynamics losses, especially acceleration/jerk statistics, path length control, and peak velocity control.
```

## 11. Update Template

새 실험을 할 때마다 아래 형식으로 추가합니다.

```text
## N. Experiment Name

모델 조합:
- ...

이전 버전 대비 변경점:
- ...

변경 이유:
- ...

기대효과:
- ...

학습 설정:
- ...

평가 결과:
- ...

해석:
- ...

개선방안:
- ...
```
