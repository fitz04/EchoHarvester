# EchoHarvester Training Module

Conformer CTC 기반 한국어 ASR 모델 훈련 모듈.
EchoHarvester 파이프라인이 생성한 Lhotse Shar 데이터를 사용하여 end-to-end ASR 모델을 훈련한다.

## 목차

- [개요](#개요)
- [전제 조건](#전제-조건)
- [빠른 시작](#빠른-시작)
- [설정](#설정)
- [CLI 명령어](#cli-명령어)
- [파이프라인 단계별 상세](#파이프라인-단계별-상세)
- [모델 아키텍처](#모델-아키텍처)
- [디렉토리 구조](#디렉토리-구조)
- [팁과 주의사항](#팁과-주의사항)

---

## 개요

```
output/shar/ (Lhotse Shar)
    ↓  train prepare
training_data/ (train/val/test splits + tokens.txt)
    ↓  train run
exp/ (checkpoints + training_stats.json)
    ↓  train decode
exp/decode_test.json (CER 평가)
    ↓  train export
exp/model.onnx 또는 model.pt (배포용 모델)
```

icefall을 import하지 않는 standalone 구현이며, 기존 의존성(`torch`, `torchaudio`, `lhotse`)만으로 동작한다.

## 전제 조건

1. EchoHarvester 파이프라인을 실행하여 `output/shar/`에 Shar 데이터가 있어야 한다.
2. PyTorch가 설치되어 있어야 한다 (CPU/CUDA/MPS 모두 지원).
3. (선택) TensorBoard 로깅을 사용하려면: `pip install tensorboard`
4. (선택) ONNX 내보내기를 사용하려면: `pip install onnxscript onnx`

## 빠른 시작

```bash
# 1. 데이터 준비 (Shar → train/val/test 분할 + 토크나이저 생성)
python -m echoharvester.main train prepare

# 2. 훈련 (기본 50 에포크)
python -m echoharvester.main train run

# 3. 테스트셋 디코딩 + CER 평가
python -m echoharvester.main train decode --checkpoint exp/best.pt

# 4. 모델 내보내기
python -m echoharvester.main train export --checkpoint exp/best.pt --format torchscript
```

## 설정

`config.yaml`의 `training:` 섹션에서 모든 훈련 파라미터를 제어한다.

### 전체 설정 예시

```yaml
training:
  # Shar 데이터 소스 (여러 경로 지정 가능)
  shar_sources:
    - "./output/shar"
    # - "//NAS-SERVER/shared/shar_data"

  data_dir: "./training_data"     # 분할된 학습 데이터 저장 경로
  exp_dir: "./exp"                # 체크포인트, 로그 저장 경로

  # 데이터 분할
  split:
    train_ratio: 0.9
    val_ratio: 0.05
    test_ratio: 0.05
    seed: 42                      # 재현성 시드

  # 토크나이저
  tokenizer:
    type: "char"                  # character-level (한글 음절 단위)

  # 오디오 특징
  features:
    num_mel_bins: 80              # Mel 필터뱅크 차원

  # 모델 아키텍처
  model:
    type: "conformer_ctc"
    attention_dim: 256            # 어텐션/임베딩 차원
    num_encoder_layers: 12        # Conformer 레이어 수
    num_attention_heads: 4        # 멀티헤드 어텐션 헤드 수
    feedforward_dim: 2048         # FFN 확장 차원
    depthwise_conv_kernel_size: 31  # ConvModule 커널 크기
    dropout: 0.1

  # 훈련 하이퍼파라미터
  training_params:
    num_epochs: 50
    max_duration: 200.0           # 배치당 최대 오디오 길이 (초)
    lr_factor: 2.5                # Noam 스케줄러 lr factor
    warm_step: 5000               # 워밍업 스텝
    weight_decay: 0.000001
    clip_grad_norm: 5.0           # 그래디언트 클리핑
    log_interval: 50              # N 배치마다 로그 출력
    valid_interval: 1             # N 에포크마다 검증
    keep_last_n: 5                # 최근 N개 체크포인트만 유지

  device: "auto"                  # auto → CUDA > MPS > CPU 순 자동 감지
```

### 주요 파라미터 설명

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `max_duration` | 배치 크기를 초 단위로 제어. 메모리가 부족하면 줄인다. | GPU 8GB: 100, 16GB: 200, 24GB+: 400 |
| `attention_dim` | 모델 크기를 결정하는 핵심 파라미터. | 소형: 256, 중형: 512 |
| `num_encoder_layers` | 레이어가 많을수록 성능↑, 속도↓. | 12 (기본), 6 (경량) |
| `warm_step` | LR 워밍업 스텝. 데이터가 적으면 줄인다. | 데이터 < 10시간: 1000, 10~100시간: 5000 |
| `keep_last_n` | 디스크 용량 관리. 체크포인트 1개 ≈ 모델 크기 × 3. | 5 |

## CLI 명령어

모든 명령은 `--config` 옵션으로 설정 파일을 지정할 수 있다 (기본: `config.yaml`).

### `train prepare`

Shar 데이터를 로드하여 train/val/test로 분할하고, character-level 토크나이저를 생성한다.

```bash
python -m echoharvester.main train prepare
```

**동작 순서:**
1. `shar_sources`의 모든 경로에서 CutSet 로드
2. Shar in-memory 오디오를 `training_data/audio/`에 WAV 파일로 저장
3. `media_id` 기준 stratified split (같은 영상의 세그먼트는 같은 split으로 → 데이터 누출 방지)
4. 훈련 텍스트에서 고유 문자 추출 → `lang_char/tokens.txt` 생성

**출력:**
```
training_data/
├── audio/                   # WAV 파일 (Shar에서 추출)
├── train_cuts.jsonl.gz      # 훈련 세트
├── val_cuts.jsonl.gz        # 검증 세트
├── test_cuts.jsonl.gz       # 테스트 세트
├── lang_char/
│   └── tokens.txt           # 토큰 사전 (icefall 포맷)
└── stats.json               # 분할 통계
```

### `train run`

Conformer CTC 모델을 훈련한다.

```bash
# 기본 실행 (config.yaml의 num_epochs만큼)
python -m echoharvester.main train run

# 에포크 수 오버라이드
python -m echoharvester.main train run --epochs 10

# 체크포인트에서 이어서 훈련
python -m echoharvester.main train run --resume exp/epoch-10.pt --epochs 50
```

**출력:**
```
exp/
├── epoch-{N}.pt             # 에포크별 체크포인트
├── best.pt                  # 최소 val_loss 체크포인트
├── training_stats.json      # 에포크별 loss/lr 기록
└── tensorboard/             # TensorBoard 이벤트 (설치 시)
```

**체크포인트 내용:**
- `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`
- `epoch`, `train_loss`, `val_loss`
- `config` (설정 스냅샷)

### `train status`

현재 훈련 상태를 확인한다.

```bash
python -m echoharvester.main train status
```

출력 예시:
```
=== Training Status ===
Best val_loss: 6.0601
Best epoch: 4
Last epoch: 4
  train_loss: 6.1311
  val_loss: 6.0601

Checkpoints (4):
  epoch-1.pt (386.7 MB)
  epoch-2.pt (386.7 MB)
  epoch-3.pt (386.7 MB)
  epoch-4.pt (386.7 MB)
  best.pt (symlink)

  Tokenizer: 432 tokens
```

### `train decode`

저장된 체크포인트로 테스트셋(또는 검증셋)을 디코딩하고 CER을 계산한다.

```bash
# 테스트셋 디코딩 (기본)
python -m echoharvester.main train decode --checkpoint exp/best.pt

# 검증셋 디코딩
python -m echoharvester.main train decode --checkpoint exp/best.pt --split val
```

**출력:** `exp/decode_test.json` (또는 `decode_val.json`)
```json
{
  "split": "test",
  "num_utterances": 100,
  "overall_cer": 0.1523,
  "samples": [
    {"ref": "안녕하세요", "hyp": "안녕하세요", "cer": 0.0},
    ...
  ]
}
```

### `train export`

훈련된 모델을 배포용 포맷으로 내보낸다.

```bash
# TorchScript (권장, 추가 의존성 없음)
python -m echoharvester.main train export --checkpoint exp/best.pt --format torchscript

# ONNX (pip install onnxscript onnx 필요)
python -m echoharvester.main train export --checkpoint exp/best.pt --format onnx

# 출력 경로 지정
python -m echoharvester.main train export --checkpoint exp/best.pt --format torchscript --output ./deploy/model.pt
```

## 파이프라인 단계별 상세

### 1단계: 데이터 준비 (`train prepare`)

**Shar 소스 탐지 로직:**
- 디렉토리에 `cuts.*.jsonl.gz` 파일이 있으면 → `CutSet.from_shar()` (오디오 포함 로드)
- 없으면 → `*.jsonl.gz`를 일반 매니페스트로 로드

**Stratified Split:**
- `media_id` 기준으로 그룹핑 (같은 영상의 모든 세그먼트가 같은 split에 배치)
- 데이터 누출(data leakage) 방지: 훈련에 사용된 영상의 세그먼트가 테스트에 나오지 않음
- 단일 소스에서 모든 데이터가 올 경우, val/test에 최소 1개 cut을 보장

**토크나이저:**
- 특수 토큰: `<blk>`(0, CTC blank), `<sos/eos>`(1), `<unk>`(2)
- 이후 Unicode 코드포인트 순서로 문자 ID 부여
- 한글 음절(AC00-D7A3)이 자연스럽게 정렬됨

### 2단계: 훈련 (`train run`)

**데이터 로딩:**
- `SimpleCutSampler`로 `max_duration` 기준 동적 배칭
- `lhotse.Fbank` 80-dim 특징 on-the-fly 추출 (디스크 절약)

**학습률 스케줄러:**
- Noam 스케줄러 (Attention Is All You Need)
- `lr = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`

**손실 함수:**
- `torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)`

**체크포인트 관리:**
- 매 에포크 저장, `keep_last_n`개만 유지 (이전 것 자동 삭제)
- `best.pt`는 최소 val_loss 기준으로 별도 저장

### 3단계: 디코딩 (`train decode`)

- CTC Greedy Decode: argmax → blank 제거 → 연속 중복 제거
- Character Error Rate (CER): 공백 제거 후 문자 단위 편집 거리

### 4단계: 내보내기 (`train export`)

- **TorchScript**: `torch.jit.trace` 기반, PyTorch만 있으면 추론 가능
- **ONNX**: `torch.onnx.export`, ONNX Runtime 등 다양한 추론 엔진에서 사용 가능

## 모델 아키텍처

```
Input: Fbank (batch, time, 80)
  │
  ▼
Conv2dSubsampling
  │  Conv2d(1, dim, 3×3, stride=2) + ReLU
  │  Conv2d(dim, dim, 3×3, stride=2) + ReLU
  │  Linear → (batch, time//4, attention_dim)
  │
  ▼
PositionalEncoding (sinusoidal)
  │
  ▼
ConformerEncoderLayer × N  (Macaron-net)
  │  ┌─ 0.5 × FeedForward (LayerNorm → Linear → SiLU → Linear)
  │  ├─ MultiHeadSelfAttention (LayerNorm → MHA → Dropout)
  │  ├─ ConvModule (LayerNorm → Pointwise → GLU → Depthwise → BN → SiLU → Pointwise)
  │  └─ 0.5 × FeedForward
  │  └─ LayerNorm
  │
  ▼
Linear(attention_dim, vocab_size)
  │
  ▼
log_softmax → CTC Loss
```

기본 설정 (attention_dim=256, layers=12): **32.9M 파라미터**

## 디렉토리 구조

```
echoharvester/training/
├── __init__.py          # 모듈 초기화
├── config.py            # Pydantic 설정 모델들
├── data_prep.py         # DataPreparer (Shar 로드, 분할)
├── tokenizer.py         # CharTokenizer (tokens.txt 생성/로드/인코딩/디코딩)
├── model.py             # ConformerCtcModel (Conv2dSubsampling, Conformer, CTC)
├── dataset.py           # AsrDataset + create_dataloader
├── utils.py             # 유틸리티 (device, seed, NoamScheduler, CER)
├── trainer.py           # Trainer (에포크 루프, 체크포인트, 검증)
├── decode.py            # Decoder (CTC greedy, CER 평가)
└── export.py            # ModelExporter (ONNX, TorchScript)
```

## 팁과 주의사항

### 메모리 관리

- `max_duration`을 줄이면 배치당 메모리 사용량이 감소한다.
- GPU 메모리 부족 시: `max_duration: 100` → `50` 으로 줄여본다.
- CPU 훈련 시 `device: "cpu"`, MPS 사용 시 `device: "mps"`.

### 데이터 양과 성능

- 120 cuts (12분): 모델 동작 검증용. 의미 있는 ASR 성능은 기대하기 어렵다.
- 10~50시간: 도메인 특화 모델로 사용 가능.
- 100시간+: 범용 한국어 ASR로 활용 가능.

### 여러 Shar 소스 합치기

```yaml
training:
  shar_sources:
    - "./output/shar"                              # 로컬
    - "/mnt/nas/project1/shar"                     # NAS
    - "//DESKTOP-I7ITVII/easystore/data/shar_data" # Windows 네트워크 공유
```

`train prepare`가 모든 소스를 자동으로 병합한다.

### TensorBoard 모니터링

```bash
pip install tensorboard
tensorboard --logdir exp/tensorboard
```

브라우저에서 `http://localhost:6006`으로 loss, learning rate 차트 확인.

### 이어서 훈련하기

```bash
# 10 에포크 훈련
python -m echoharvester.main train run --epochs 10

# 이어서 50 에포크까지
python -m echoharvester.main train run --resume exp/epoch-10.pt --epochs 50
```

`--resume`은 모델 가중치, 옵티마이저 상태, 스케줄러 스텝을 모두 복원한다.

### 토크나이저 형식

`tokens.txt`는 icefall과 동일한 포맷:
```
<blk> 0
<sos/eos> 1
<unk> 2
, 3
. 4
가 20
나 21
...
```

Python에서 직접 사용:
```python
from echoharvester.training.tokenizer import CharTokenizer
tok = CharTokenizer(Path("training_data/lang_char/tokens.txt"))
ids = tok.encode("안녕하세요")    # [283, 305, 370, 156, 416]
text = tok.decode(ids)            # "안녕하세요"
```
