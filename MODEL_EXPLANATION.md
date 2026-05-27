# 마우스 궤적 생성 모델 정리

이 문서는 `data_re.py`와 `mouse_trajectory_model.py`의 역할, 실행 방법, 딥러닝 모델 내부 구조, 그리고 `CVAE + GRU`를 사용한 이유를 정리한 설명입니다.

## 전체 흐름

```text
원본 XML
  ↓
data_re.py
  ↓
normalized_output/normalized_trajectory_points.csv
  ↓
mouse_trajectory_model.py train
  ↓
models/mouse_trajectory_cvae.pt
  ↓
mouse_trajectory_model.py generate
  ↓
generated/trajectories.csv
```

## 파일 역할

### `data_re.py`

`data_re.py`는 원본 XML 마우스 이동 데이터를 읽어서 딥러닝 학습에 사용할 수 있는 정규화 CSV로 바꾸는 전처리 코드입니다.

이 스크립트는 실제 화면 좌표를 그대로 쓰지 않고, 모든 마우스 이동을 같은 기준 좌표계로 변환합니다.

```text
시작점 = (0, 0)
목표점 = (1, 0)
```

즉 화면에서 어디에서 어디로 움직였든, 모델 입장에서는 항상 시작점에서 목표점으로 가는 표준 궤적으로 보게 됩니다.

정규화 결과에는 주로 다음 값들이 들어갑니다.

```text
x_norm    정규화된 x 좌표
y_norm    정규화된 y 좌표
t_norm    정규화된 시간
A         이동 거리
W         목표 크기
ID        Fitts' law 난이도
duration  이동 시간
error     실패 trial 여부
```

### `mouse_trajectory_model.py`

`mouse_trajectory_model.py`는 정규화된 CSV를 이용해 딥러닝 모델을 학습하고, 학습된 모델로 새로운 마우스 궤적을 생성하는 코드입니다.

즉 역할을 나누면 다음과 같습니다.

```text
data_re.py = 정규화 / 전처리
mouse_trajectory_model.py = 딥러닝 학습 / 궤적 생성
```

## 실행 방법

먼저 필요한 패키지를 설치합니다.

```powershell
pip install -r requirements.txt
```

원본 XML 데이터를 정규화합니다.

```powershell
python data_re.py
```

정규화된 CSV는 기본적으로 다음 위치에 저장됩니다.

```text
normalized_output/normalized_trajectory_points.csv
```

모델을 학습합니다.

```powershell
python mouse_trajectory_model.py train --csv normalized_output/normalized_trajectory_points.csv
```

학습된 모델은 기본적으로 다음 위치에 저장됩니다.

```text
models/mouse_trajectory_cvae.pt
```

새 궤적을 생성합니다.

```powershell
python mouse_trajectory_model.py generate `
  --A 500 --W 40 --ID 3.75 --duration 650 `
  --start-x 200 --start-y 300 `
  --target-x 700 --target-y 450 `
  --count 5
```

생성 결과는 기본적으로 다음 위치에 저장됩니다.

```text
generated/trajectories.csv
```

결과 CSV에는 다음 좌표가 함께 들어갑니다.

```text
x_norm, y_norm  정규화 좌표
x, y            실제 화면 좌표
```

## 입력 데이터가 모델에 들어가는 방식

CSV 안의 여러 점들은 하나의 trial 단위로 다시 묶입니다.

기준은 다음 값들입니다.

```python
GROUP_KEYS = ("subject", "subfolder", "condition_index", "trial_number")
```

같은 사람, 같은 조건, 같은 trial 번호에 속한 점들을 하나의 마우스 이동 궤적으로 봅니다.

각 궤적은 길이가 서로 다를 수 있습니다. 어떤 궤적은 20개 점이고, 어떤 궤적은 120개 점일 수 있습니다. 모델에 넣기 위해 모든 궤적을 같은 길이로 다시 샘플링합니다.

기본 길이는 다음과 같습니다.

```text
seq_len = 64
```

즉 모든 궤적은 64개의 `(x_norm, y_norm)` 점으로 변환됩니다.

## 모델이 보는 조건값

모델은 궤적만 보는 것이 아니라 조건값도 함께 봅니다.

```python
COND_KEYS = ("A", "W", "ID", "duration")
```

각 값의 의미는 다음과 같습니다.

```text
A         이동 거리
W         목표 크기
ID        Fitts' law index of difficulty
duration  이동 시간
```

모델은 이런 질문에 답하도록 학습됩니다.

```text
거리 A가 500이고,
목표 크기 W가 40이고,
난이도 ID가 3.75이고,
이동 시간이 650ms 정도라면,
사람 마우스 궤적은 보통 어떻게 생겼을까?
```

## 모델 구조

핵심 모델은 `TrajectoryCVAE`입니다.

```text
CVAE = Conditional Variational AutoEncoder
GRU = Gated Recurrent Unit
```

전체 구조는 다음과 같습니다.

```text
조건 + 실제 궤적
  ↓
Encoder GRU
  ↓
잠재공간 z의 평균과 분산
  ↓
랜덤 샘플 z
  ↓
Decoder GRU
  ↓
생성된 궤적
```

### Encoder

Encoder는 실제 사람 궤적과 조건값을 함께 입력받아 잠재공간의 분포를 만듭니다.

입력은 다음과 같습니다.

```text
traj = 64개짜리 (x_norm, y_norm) 시퀀스
cond = A, W, ID, duration
```

각 시점마다 Encoder가 보는 값은 다음과 같습니다.

```text
x_norm, y_norm, A, W, ID, duration
```

Encoder는 GRU를 통해 전체 궤적의 흐름을 읽고, 최종적으로 두 값을 만듭니다.

```text
mu      잠재벡터 평균
logvar  잠재벡터 분산의 로그값
```

### Reparameterization

CVAE는 `mu`와 `logvar`에서 실제 잠재벡터 `z`를 샘플링합니다.

```python
std = torch.exp(0.5 * logvar)
z = mu + torch.randn_like(std) * std
```

이 과정 덕분에 같은 조건을 넣어도 매번 조금씩 다른 궤적을 생성할 수 있습니다.

### Decoder

Decoder는 조건값과 랜덤 잠재벡터 `z`, 그리고 시간 위치 `t`를 받아 궤적을 생성합니다.

각 시점마다 Decoder가 보는 값은 다음과 같습니다.

```text
A, W, ID, duration, z, t
```

출력은 다음과 같습니다.

```text
64개의 (x_norm, y_norm)
```

생성된 궤적의 첫 점과 마지막 점은 강제로 고정됩니다.

```text
첫 점 = (0, 0)
마지막 점 = (1, 0)
```

그래서 생성된 궤적은 반드시 시작점에서 출발해서 목표점에 도착합니다. 중간 경로만 모델이 자유롭게 만듭니다.

## 학습 손실

학습할 때 사용하는 손실은 세 가지를 합친 형태입니다.

```python
loss = recon + beta * kld + smooth_weight * smoothness_loss
```

각 항의 의미는 다음과 같습니다.

```text
recon
실제 사람 궤적과 생성 궤적이 얼마나 비슷한지 측정

kld
잠재공간 z가 너무 제멋대로 퍼지지 않도록 정리

smoothness_loss
궤적이 너무 각지거나 덜컥거리지 않도록 부드러움 유도
```

`smoothness_loss`는 궤적의 가속도 변화를 줄이는 방식입니다.

```python
velocity = traj[:, 1:] - traj[:, :-1]
acceleration = velocity[:, 1:] - velocity[:, :-1]
smoothness_loss = acceleration.pow(2).mean()
```

사람의 마우스 움직임은 보통 순간적으로 크게 꺾이기보다는 연속적이고 부드럽기 때문에 이 항이 중요합니다.

## 생성 과정

생성할 때는 학습된 모델을 불러온 뒤 조건값을 넣습니다.

```text
A, W, ID, duration
```

그리고 랜덤 잠재벡터 `z`를 만듭니다.

```python
z = torch.randn(count, latent_dim) * temperature
```

`temperature`는 생성 다양성을 조절합니다.

```text
temperature 낮음  → 더 평균적이고 안정적인 궤적
temperature 높음  → 더 다양하고 흔들림 있는 궤적
```

모델은 먼저 정규화 좌표를 생성합니다.

```text
(0, 0)에서 시작해서 (1, 0)에 도착하는 궤적
```

그 다음 `normalized_to_screen` 함수가 이 정규화 좌표를 실제 화면 좌표로 변환합니다.

개념적으로는 다음과 같습니다.

```text
정규화 궤적
  ↓
실제 거리만큼 확대
  ↓
시작점에서 목표점 방향으로 회전
  ↓
화면 좌표로 이동
```

예를 들어 시작점이 `(200, 300)`이고 목표점이 `(700, 450)`이면, 생성된 정규화 궤적이 그 방향과 거리로 변환됩니다.

## 왜 CVAE를 사용했는가

마우스 궤적은 정답이 하나가 아닙니다.

같은 시작점, 목표점, 거리, 목표 크기, 이동 시간이 있어도 사람마다 경로가 다릅니다. 같은 사람도 매번 완전히 같은 경로로 움직이지 않습니다.

예를 들어 어떤 사람은 거의 직선으로 갈 수 있고, 어떤 사람은 살짝 위로 휘었다가 갈 수 있고, 어떤 사람은 목표 근처에서 조금 흔들릴 수 있습니다.

그래서 이 문제는 다음과 같은 문제에 가깝습니다.

```text
하나의 정답 궤적 예측
```

보다는:

```text
사람이 만들 법한 궤적 분포에서 하나를 생성
```

CVAE는 이 목적에 잘 맞습니다.

CVAE를 사용한 이유는 다음과 같습니다.

```text
조건부 생성이 가능함
같은 조건에서도 다양한 결과를 만들 수 있음
데이터의 분포를 학습하기 좋음
GAN보다 학습이 안정적인 편임
```

여기서 조건부 생성이란 다음 조건을 보고 궤적을 만드는 것을 뜻합니다.

```text
A, W, ID, duration
```

## 왜 GRU를 사용했는가

마우스 궤적은 순서가 있는 데이터입니다.

```text
점1 → 점2 → 점3 → ... → 점64
```

각 점은 독립적이지 않습니다. 현재 위치는 이전 위치와 자연스럽게 이어져야 합니다.

GRU는 이런 시계열 데이터를 처리하는 모델입니다. 이전 상태를 기억하면서 다음 흐름을 만들 수 있습니다.

GRU를 사용한 이유는 다음과 같습니다.

```text
마우스 궤적이 시간 순서 데이터라서
이전 점과 다음 점의 연속성을 배우기 좋아서
LSTM보다 구조가 가볍고 빠른 편이라서
Transformer보다 작은 데이터셋에서 부담이 적어서
```

## 왜 CVAE와 GRU를 같이 쓰는가

둘의 역할은 서로 다릅니다.

```text
CVAE
이 조건에서 어떤 스타일의 궤적이 나올 수 있는지 학습
예: 직선형, 살짝 휘는형, 흔들림 많은형

GRU
그 스타일을 시간 순서대로 자연스럽게 이어지는 점들로 생성
```

한 줄로 정리하면 다음과 같습니다.

```text
CVAE = 다양성과 조건부 생성
GRU = 시간적 연속성과 자연스러운 흐름
```

즉 이 프로젝트에서 `CVAE + GRU`를 사용한 이유는 마우스 궤적 생성이 다음 문제이기 때문입니다.

```text
조건부 랜덤 시계열 생성 문제
```

## 다른 모델과 비교

### MLP를 쓰지 않은 이유

MLP로도 64개 점을 한 번에 예측할 수는 있습니다.

하지만 MLP는 궤적을 큰 벡터로 봅니다.

```text
[x1, y1, x2, y2, ..., x64, y64]
```

이렇게 보면 시간적으로 이어진 움직임이라는 구조를 충분히 활용하기 어렵습니다.

GRU는 앞 점과 뒤 점의 흐름을 순서대로 다루기 때문에 궤적 데이터에 더 자연스럽습니다.

### GAN을 쓰지 않은 이유

GAN도 사람 같은 궤적 생성에 사용할 수 있습니다.

하지만 GAN은 학습이 예민한 편입니다.

```text
generator와 discriminator 균형 문제
mode collapse 문제
학습 불안정 문제
```

처음 안정적으로 동작하는 생성 모델을 만들기에는 CVAE가 더 실용적입니다.

### Transformer를 쓰지 않은 이유

Transformer도 사용할 수 있습니다.

하지만 이 데이터는 보통 궤적 길이가 짧고, 데이터 규모가 아주 크지 않을 가능성이 큽니다.

Transformer는 강력하지만 데이터와 연산 비용이 더 많이 필요합니다.

GRU는 짧은 시퀀스에는 충분히 강하고 가볍습니다.

## 최종 요약

```text
data_re.py
원본 XML을 정규화 CSV로 변환한다.

mouse_trajectory_model.py
정규화된 CSV를 이용해 사람 같은 마우스 궤적 생성 모델을 학습하고 생성한다.

CVAE
조건에 맞는 다양한 궤적 분포를 학습한다.

GRU
궤적의 시간적 흐름과 자연스러운 연속성을 학습한다.
```

가장 중요한 핵심은 다음과 같습니다.

```text
data_re.py는 좌표계를 통일하고,
mouse_trajectory_model.py는 사람 움직임의 분포를 학습한다.
```
