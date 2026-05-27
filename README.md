# mousePage

Fitts task XML 데이터를 정규화하고, 정규화된 마우스 궤적으로 사람과 유사한 이동 경로를 생성하는 CVAE+GRU 모델입니다.

## 1. 정규화

원본 데이터가 `fittsdata/P*/[2-8]/s01_2D_nomet__*.xml` 구조로 있을 때:

```powershell
python data_re.py
```

결과는 `normalized_output/normalized_trajectory_points.csv`에 저장됩니다.

## 2. 학습

```powershell
pip install -r requirements.txt
python mouse_trajectory_model.py train --csv normalized_output/normalized_trajectory_points.csv
```

모델은 기본적으로 `models/mouse_trajectory_cvae.pt`에 저장됩니다.

## 3. 궤적 생성

```powershell
python mouse_trajectory_model.py generate `
  --A 500 --W 40 --ID 3.75 --duration 650 `
  --start-x 200 --start-y 300 `
  --target-x 700 --target-y 450 `
  --count 5
```

결과는 `generated/trajectories.csv`에 저장됩니다. `x_norm`, `y_norm`은 정규화 좌표이고 `x`, `y`는 실제 화면 좌표입니다.
