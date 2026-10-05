# 적군·아군 YOLO 학습 구조

## 1. 목표

카메라 영상에서 표적을 탐지하고, 각 bounding box를 다음 두 클래스 중 하나로 분류한다.

| Class ID | 이름 | 용도 |
|---:|---|---|
| 0 | `enemy` | 추적 및 조준 후보 |
| 1 | `ally` | 탐지만 수행하며 조준 대상에서 제외 |

이 학습은 이미지 전체를 하나의 클래스로 분류하는 binary image classification이 아니라, 이미지 안의 여러 객체를 각각 찾고 분류하는 **2-class object detection**이다.

## 2. 전체 파이프라인

```text
학습 이미지 + YOLO bounding-box 라벨
                 │
                 ▼
        YOLO11n 사전학습 가중치
                 │
                 ▼
       2-class detection fine-tuning
                 │
                 ▼
 runs/detect/targets_combined_v2/
 ├─ weights/best.pt
 ├─ weights/last.pt
 ├─ results.csv
 ├─ confusion_matrix.png
 └─ 각종 학습/검증 그래프
```

실제 운용 단계에서는 YOLO가 검출한 객체 중 `class_id == 0`인 enemy만 선택한다. 선택한 enemy의 bounding-box 중심이 화면 중앙에 올 때까지 모터를 움직인 다음, 여러 프레임에서 중앙 정렬과 enemy 판정이 유지될 때만 발사 조건을 만족하도록 구성한다.

## 3. 데이터셋

학습 설정 파일은 `data/targets_combined.yaml`이다.

```yaml
path: E:/MK/junior/physicalAI/data
train:
  - dataset_90_5_5/images/train
  - yolo_color-3-yolov11-png-class-order/train/images
val:
  - dataset_90_5_5/images/val
  - yolo_color-3-yolov11-png-class-order/valid/images
test:
  - dataset_90_5_5/images/test
  - yolo_color-3-yolov11-png-class-order/test/images
names:
  0: enemy
  1: ally
```

현재 분포는 다음과 같다.

| Split | 이미지 | enemy box | ally box |
|---|---:|---:|---:|
| train | 933 | 1,112 | 1,119 |
| validation | 59 | 71 | 74 |
| test | 59 | 70 | 70 |

클래스 개수는 균형적이다. 다만 대부분의 이미지에 enemy와 ally가 함께 등장하고 배경-only 이미지가 없으므로, 이후 단일 클래스 장면과 표적이 없는 배경 데이터를 추가하는 것이 권장된다.

원본 데이터셋의 클래스 순서는 서로 달랐다. `prepare_targets_dataset.py`가 JPG 데이터셋의 class ID를 뒤집어 최종 기준을 `0=enemy`, `1=ally`로 통일한다. 이미 생성된 원본 및 변환 데이터셋은 학습 중 변경하지 않는다.

## 4. 모델 구조

기본 모델은 COCO 사전학습 가중치가 적용된 `yolo11n.pt`이다. Nano 모델은 라즈베리파이 배포를 고려해 속도와 정확도 사이의 균형을 맞춘 선택이다.

```text
입력 이미지 (960 × 960)
        │
        ▼
Backbone
  Conv + C3k2 블록으로 저수준/고수준 특징 추출
        │
        ▼
SPPF + C2PSA
  여러 크기의 문맥과 중요한 특징 결합
        │
        ▼
Neck
  Upsample + Concat + C3k2
  작은 객체와 큰 객체용 다중 스케일 특징 결합
        │
        ▼
Detection Head
  3개 스케일에서 bounding box와 class score 출력
        │
        ├─ box 좌표
        ├─ confidence
        └─ class: enemy / ally
```

YOLO11n은 약 260만 개의 파라미터를 가진 경량 detection 모델이다. 표적 크기가 이미지에서 매우 작기 때문에 학습 입력 크기는 기본 640보다 큰 960을 사용한다.

## 5. 학습 설정

학습 진입점은 `train_targets.py`이다.

| 설정 | 값 | 목적 |
|---|---:|---|
| pretrained model | `yolo11n.pt` | COCO 특징을 이용한 fine-tuning |
| epochs | 80 | 데이터 규모를 고려한 충분한 반복 |
| image size | 960 | 작은 표적 보존 |
| batch | 8 | RTX 4070 12GB 기준 |
| patience | 20 | validation 개선이 없을 때 조기 종료 |
| device | 0 | 첫 번째 CUDA GPU |
| seed | 42 | 재현성 확보 |
| checkpoint | 10 epoch 간격 | 중단 시 복구 지원 |

적군과 아군이 색상으로 구분되므로 기본 YOLO 색상 증강보다 약하게 설정한다.

| 증강 | 값 |
|---|---:|
| hue | 0.0 |
| saturation | 0.15 |
| brightness/value | 0.15 |
| horizontal flip | 0.5 |
| vertical flip | 0.0 |
| 마지막 mosaic 비활성화 구간 | 10 epochs |

## 6. 손실과 평가

학습 과정에서는 다음 주요 손실을 최적화한다.

- Box loss: 예측 bounding box와 정답 box의 위치 차이
- Classification loss: enemy와 ally 분류 오차
- DFL loss: bounding box 경계 분포 회귀 오차

최종 판단에서는 전체 mAP만 보지 않고 클래스별 결과를 확인한다.

- enemy precision/recall
- ally precision/recall
- mAP50 및 mAP50-95
- confusion matrix
- ally를 enemy로 잘못 분류한 횟수
- 배경에서 발생하는 false positive

발사 시스템에서는 특히 **ally → enemy 오분류**가 가장 중요한 실패 유형이다. 실제 적용 전 별도의 카메라 영상으로 반드시 검증해야 한다.

## 7. 실행 환경 및 명령

모든 Python 라이브러리는 프로젝트의 `.venv`에서 사용한다.

```powershell
.\.venv\Scripts\python.exe .\train_targets.py
```

최종 가중치는 다음 경로에 생성된다.

```text
runs/detect/targets_combined_v2/weights/best.pt
```

`best.pt`는 validation 성능이 가장 좋았던 epoch의 모델이고, `last.pt`는 가장 마지막 epoch의 모델이다. 실제 평가와 배포에는 원칙적으로 `best.pt`를 사용한다.

## 8. 실제 추론 시 필수 조건

현재 `inference.py`는 가장 confidence가 높은 객체를 클래스와 관계없이 선택하므로, 2-class 모델 배포 전에 수정해야 한다.

필수 동작은 다음과 같다.

1. `class_id == 0`인 enemy만 조준 후보로 선택한다.
2. ally는 표시만 하고 모터 목표로 사용하지 않는다.
3. enemy confidence 임계값을 적용한다.
4. enemy 중심과 화면 중심 사이의 오차로 모터를 제어한다.
5. 중앙 정렬이 여러 프레임 연속 유지된 경우에만 조준 완료로 판단한다.
6. ally/enemy 판정이 불안정하거나 표적이 사라지면 발사하지 않는다.
7. YOLO에는 충분한 해상도의 프레임을 입력하고, YawCNN용 224 × 224 입력과 분리한다.

## 9. `targets_combined_v2` 학습 결과

2026-10-05에 RTX 4070과 프로젝트 `.venv`를 사용해 80 epochs 학습을 완료했다. validation의 `mAP50-95`가 가장 높았던 epoch는 70이다.

### Validation (`best.pt`)

| 범위 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 전체 | 0.981 | 0.904 | 0.967 | 0.792 |
| enemy | 0.985 | 0.931 | 0.982 | 0.820 |
| ally | 0.977 | 0.878 | 0.952 | 0.765 |

### Test (`best.pt`)

| 범위 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 전체 | 0.898 | 0.936 | 0.958 | 0.788 |
| enemy | 0.915 | 0.929 | 0.954 | 0.786 |
| ally | 0.880 | 0.943 | 0.962 | 0.790 |

학습에 약 0.349시간이 소요됐다. test precision이 validation보다 낮으므로 실제 카메라 환경에서 confidence threshold와 오분류를 추가 검증해야 한다.

최종 모델:

```text
runs/detect/targets_combined_v2/weights/best.pt
```

학습 그래프와 validation confusion matrix는 `runs/detect/targets_combined_v2/`에, test confusion matrix와 예측 이미지는 `runs/detect/val/`에 저장되어 있다.
