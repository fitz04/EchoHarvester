# EchoHarvester

한국어 ASR(자동 음성 인식) 훈련 데이터 파이프라인

유튜브 및 로컬 미디어 파일에서 한국어 자막/음성을 수집하여 고신뢰도 ASR 학습 데이터를 생성합니다.

## 주요 기능

- **다양한 입력 소스 지원**
  - YouTube 채널/플레이리스트/단일 영상
  - 로컬 동영상/음성 파일
  - 디렉토리 일괄 처리

- **5단계 파이프라인**
  1. 메타데이터 수집
  2. 다운로드/준비
  3. CPU 전처리 (강제 정렬, 자막 파싱, 정규화, SNR/VAD 필터링)
  4. GPU 검증 (Qwen3-ASR 재전사, CER 필터링)
  5. Lhotse Shar 내보내기

- **강제 정렬 (Forced Alignment)**
  - Qwen3-ForcedAligner-0.6B 기반 음성-텍스트 정밀 타임스탬프 정렬
  - VAD 기반 청킹: Silero VAD로 침묵 구간을 감지하여 자연스러운 경계에서 오디오 분할
  - 구두점 정규화 매칭: FA 출력과 원본 자막 간 구두점 차이를 제거하여 정확한 문자열 매칭
  - 15분 영상 기준 테스트 결과: gpu_pass 16개 → **149개** (9.3배 향상)

- **WebUI**
  - 대시보드: 통계 및 진행 현황
  - 소스 관리: 입력 소스 추가/삭제
  - 파이프라인 제어: 시작/중지/개별 스테이지 실행
  - 데이터 탐색: 처리된 세그먼트 검색/필터

## 설치

```bash
# 저장소 클론
git clone https://github.com/fitz04/EchoHarvester.git
cd EchoHarvester

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -e .

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

### 추가 요구사항

- **ffmpeg**: 오디오 처리용
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg
  # macOS
  brew install ffmpeg
  ```

- **CUDA** (선택): GPU 가속을 위해 CUDA 지원 PyTorch 필요
  ```bash
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- **macOS (Apple Silicon)**: CUDA 미지원 환경에서도 동작
  - `faster-whisper`는 CTranslate2 기반으로 MPS(Apple GPU) 미지원
  - `device: "auto"` 설정 시 자동으로 CPU 폴백 (int8 양자화 적용)
  - CPU 모드에서도 Whisper medium 모델 사용 가능 (속도는 느려짐)
  ```yaml
  validation:
    device: "auto"        # 자동 감지 (CUDA 없으면 CPU)
    compute_type: "auto"  # 자동 설정 (CPU: int8, CUDA: float16)
  ```

## 사용법

### 1. 설정 파일 작성

`config.yaml` 파일을 편집하여 입력 소스와 설정을 구성합니다:

```yaml
sources:
  - type: youtube_channel
    url: "https://www.youtube.com/@channel_name"
    label: "뉴스채널"

  - type: local_directory
    path: "/data/videos"
    pattern: "*.mp4"
    label: "로컬영상"

filters:
  cer_threshold_manual: 0.15    # 수동자막 CER 임계값
  cer_threshold_auto: 0.10      # 자동자막 CER 임계값

validation:
  model: "seastar105/whisper-medium-komixv2"
  device: "cuda"
```

### 2. CLI 사용

```bash
# 전체 파이프라인 실행
python -m echoharvester run --config config.yaml

# 특정 스테이지만 실행
python -m echoharvester run --config config.yaml --stage preprocess

# 상태 확인
python -m echoharvester status --config config.yaml

# 에러 목록 조회
python -m echoharvester errors --config config.yaml

# 통계 리포트 생성
python -m echoharvester report --config config.yaml
```

### 3. WebUI 사용

```bash
# 웹 서버 실행
python -m echoharvester server --config config.yaml

# 브라우저에서 접속
# http://127.0.0.1:8000
```

## 프로젝트 구조

```
echoharvester/
├── config.py           # 설정 관리
├── db.py               # SQLite 데이터베이스
├── main.py             # CLI 엔트리포인트
├── sources/            # 입력 소스 구현
│   ├── youtube.py      # YouTube 소스
│   ├── local_file.py   # 로컬 파일 소스
│   └── local_directory.py
├── stages/             # 파이프라인 스테이지
│   ├── stage1_metadata.py
│   ├── stage2_download.py
│   ├── stage3_preprocess.py
│   ├── stage4_validate.py
│   └── stage5_export.py
├── utils/              # 유틸리티 모듈
│   ├── forced_alignment.py  # Qwen3-ForcedAligner 강제 정렬
│   ├── subtitle_parser.py
│   ├── text_normalize.py
│   ├── audio_utils.py
│   ├── cer.py
│   └── vad.py
├── pipeline/           # 오케스트레이터
│   └── orchestrator.py
├── api/                # FastAPI 백엔드
│   └── routes/
└── web/                # WebUI 템플릿
    └── templates/
```

## 출력 형식

처리된 데이터는 Lhotse Shar 형식으로 출력됩니다:

```
output/
├── shar/               # Lhotse Shar 아카이브
│   ├── cuts.000000.tar
│   ├── cuts.000001.tar
│   └── ...
├── manifest.jsonl.gz   # CutSet 매니페스트
└── stats.json          # 통계 정보
```

## 필터링 기준

- **길이**: 0.5초 ~ 30초
- **SNR**: 최소 10dB
- **음성 비율**: 최소 50% (VAD 기준)
- **CER**: 수동자막 15%, 자동자막 10% 이하

## 라이선스

MIT License
