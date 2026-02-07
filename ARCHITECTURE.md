# EchoHarvester - 한국어 ASR 훈련 데이터 파이프라인

## 개요

유튜브 및 로컬 미디어 파일에서 한국어 자막/음성을 수집하여 고신뢰도 ASR 학습 데이터를 생성하는 파이프라인.
WebUI를 통한 편리한 관리와 실시간 모니터링 지원.

## 핵심 설계 원칙

1. **범용 입력 소스**: YouTube, 로컬 동영상/음성 파일 모두 지원
2. **비동기 스테이지 파이프라인**: 각 단계를 큐로 연결, CPU/GPU 작업 병렬화
3. **Early Rejection**: GPU 전에 CPU로 최대한 필터링하여 GPU 부하 절감
4. **중단/재개**: SQLite로 파일 단위 처리 상태 기록
5. **에러 내성**: 개별 파일 에러 시 스킵+로깅, 전체 파이프라인 계속 진행
6. **WebUI**: 로컬 웹 인터페이스로 편리한 관리 및 모니터링

## 프로젝트 구조

```
echoharvester/
├── __init__.py
├── main.py                     # CLI 엔트리포인트
├── config.py                   # 설정 관리 (Pydantic)
├── db.py                       # SQLite 상태 관리
│
├── sources/                    # 입력 소스 추상화
│   ├── __init__.py
│   ├── base.py                 # BaseSource 추상 클래스
│   ├── youtube.py              # YouTube 채널/플레이리스트
│   ├── local_file.py           # 로컬 파일 (동영상/음성)
│   └── local_directory.py      # 디렉토리 일괄 처리
│
├── stages/                     # 파이프라인 스테이지
│   ├── __init__.py
│   ├── base.py                 # BaseStage 추상 클래스
│   ├── stage1_metadata.py      # 메타데이터 수집
│   ├── stage2_download.py      # 다운로드 (YouTube) / 복사 (로컬)
│   ├── stage3_preprocess.py    # CPU 전처리
│   ├── stage4_validate.py      # GPU 검증 (faster-whisper)
│   └── stage5_export.py        # Lhotse Shar 패킹
│
├── utils/                      # 유틸리티 모듈
│   ├── __init__.py
│   ├── subtitle_parser.py      # VTT/SRT 파싱
│   ├── text_normalize.py       # 한국어 전사 정규화
│   ├── audio_utils.py          # 오디오 처리 (ffmpeg, SNR)
│   ├── cer.py                  # CER 계산
│   └── vad.py                  # Silero VAD 래퍼
│
├── pipeline/                   # 파이프라인 코어
│   ├── __init__.py
│   ├── orchestrator.py         # 파이프라인 오케스트레이터
│   ├── queue_manager.py        # 비동기 큐 관리
│   └── worker.py               # 워커 프로세스
│
├── api/                        # FastAPI 백엔드
│   ├── __init__.py
│   ├── app.py                  # FastAPI 앱
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── sources.py          # 소스 관리 API
│   │   ├── pipeline.py         # 파이프라인 제어 API
│   │   ├── dashboard.py        # 대시보드 데이터 API
│   │   └── websocket.py        # 실시간 업데이트
│   └── deps.py                 # 의존성 주입
│
└── web/                        # WebUI 프론트엔드
    ├── static/
    │   ├── css/
    │   └── js/
    └── templates/
        ├── base.html
        ├── dashboard.html
        ├── sources.html
        └── pipeline.html

config.yaml                     # 사용자 설정 파일
requirements.txt                # Python 의존성
pyproject.toml                  # 프로젝트 메타데이터
```

## 파이프라인 흐름

```
[입력 소스]
    ├── YouTube: 채널/플레이리스트 URL
    ├── 로컬 파일: 동영상/음성 + 자막(선택)
    └── 로컬 디렉토리: 일괄 처리
           ↓
[Stage 1: 메타데이터 수집]
    - YouTube: yt-dlp로 영상 목록 + 자막 정보
    - 로컬: 파일 스캔 + 자막 파일 매칭
           ↓
[Stage 2: 다운로드/준비]
    - YouTube: 오디오(16kHz WAV) + 자막(VTT) 다운로드
    - 로컬: 오디오 변환 (필요시) + 자막 로드
           ↓ Queue A
[Stage 3: CPU 전처리]
    - 자막 파싱 및 정규화
    - 오디오 세그먼트 분할
    - SNR/VAD 기반 필터링
           ↓ Queue B
[Stage 4: GPU 검증]
    - faster-whisper로 재전사
    - CER 계산 및 필터링
           ↓ Queue C
[Stage 5: Lhotse 내보내기]
    - CutSet 구성
    - Shar 포맷 저장
    - 통계 리포트
```

## 입력 소스 추상화

### BaseSource (추상 클래스)
```python
class BaseSource(ABC):
    source_type: str
    label: str

    @abstractmethod
    async def discover(self) -> List[MediaItem]:
        """미디어 아이템 목록 반환"""
        pass

    @abstractmethod
    async def prepare(self, item: MediaItem, work_dir: Path) -> PreparedMedia:
        """오디오 + 자막 준비 (다운로드 또는 변환)"""
        pass
```

### 소스 타입별 구현
| 타입 | 설명 | 자막 처리 |
|------|------|----------|
| `youtube_channel` | YouTube 채널 전체 | 자동/수동 자막 다운로드 |
| `youtube_playlist` | YouTube 플레이리스트 | 자동/수동 자막 다운로드 |
| `youtube_video` | YouTube 단일 영상 | 자동/수동 자막 다운로드 |
| `local_file` | 로컬 미디어 파일 | 동일 경로 자막 자동 매칭 |
| `local_directory` | 디렉토리 일괄 | 파일별 자막 자동 매칭 |

### 로컬 파일 자막 매칭 규칙
```
video.mp4 → video.ko.vtt, video.ko.srt, video.vtt, video.srt
```

## WebUI 설계

### 기술 스택
- **백엔드**: FastAPI + SQLite
- **프론트엔드**: Jinja2 템플릿 + Tailwind CSS + Alpine.js + HTMX
- **실시간**: WebSocket (진행상황 스트리밍)

### 주요 페이지

#### 1. 대시보드 (`/`)
- 전체 통계 (총 영상, 세그먼트, 시간, 통과율)
- 실시간 진행 상태
- CER 분포 차트
- 최근 처리 항목

#### 2. 소스 관리 (`/sources`)
- 소스 목록 (YouTube URL, 로컬 경로)
- 소스 추가/수정/삭제
- 소스별 상태 (pending, processing, done)

#### 3. 파이프라인 제어 (`/pipeline`)
- 시작/중지/재개 버튼
- 스테이지별 진행 상황
- 실시간 로그 스트리밍
- 워커 상태 모니터링

#### 4. 데이터 탐색 (`/explore`)
- 처리된 세그먼트 검색/필터
- 샘플 오디오 재생
- 원본/정규화/재전사 텍스트 비교

### API 엔드포인트

```
# 소스 관리
GET    /api/sources              - 소스 목록
POST   /api/sources              - 소스 추가
DELETE /api/sources/{id}         - 소스 삭제

# 파이프라인 제어
POST   /api/pipeline/start       - 파이프라인 시작
POST   /api/pipeline/stop        - 파이프라인 중지
GET    /api/pipeline/status      - 현재 상태

# 대시보드
GET    /api/dashboard/stats      - 전체 통계
GET    /api/dashboard/recent     - 최근 처리 항목

# 실시간
WS     /ws/progress              - 진행상황 스트리밍
WS     /ws/logs                  - 로그 스트리밍
```

## 설정 파일 구조 (config.yaml)

```yaml
# 입력 소스
sources:
  - type: youtube_channel
    url: "https://www.youtube.com/@channel_name"
    label: "뉴스채널A"

  - type: youtube_playlist
    url: "https://www.youtube.com/playlist?list=PL..."
    label: "KBS다큐"

  - type: local_directory
    path: "/data/raw_videos"
    pattern: "*.mp4"
    label: "로컬영상"

# 자막 설정
subtitles:
  languages: ["ko"]
  include_auto_generated: true
  prefer_manual: true

# 필터링 임계값
filters:
  min_duration_sec: 0.5
  max_duration_sec: 30.0
  min_snr_db: 10.0
  min_speech_ratio: 0.5
  cer_threshold_manual: 0.15
  cer_threshold_auto: 0.10

# 다운로드 설정 (YouTube)
download:
  max_concurrent: 3
  rate_limit: "1M"

# 오디오 설정
audio:
  sample_rate: 16000
  format: "wav"
  segment_padding_sec: 0.15

# GPU 검증
validation:
  model: "seastar105/whisper-medium-komixv2"
  device: "cuda"
  compute_type: "float16"
  batch_size: 16

# 경로
paths:
  work_dir: "./work"
  output_dir: "./output"
  archive_dir: "./archive"

# 파이프라인 설정
pipeline:
  num_cpu_workers: 4
  gpu_queue_size: 100
  checkpoint_interval: 50

# 웹 서버
web:
  host: "127.0.0.1"
  port: 8000
```

## SQLite 스키마

```sql
-- 입력 소스
CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,           -- youtube_channel, local_file, etc.
    url_or_path TEXT NOT NULL,
    label TEXT,
    config_json TEXT,             -- 소스별 추가 설정
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 미디어 아이템 (영상/음성 파일)
CREATE TABLE media_items (
    id TEXT PRIMARY KEY,          -- YouTube: video_id, 로컬: 해시
    source_id INTEGER REFERENCES sources(id),
    title TEXT,
    duration_sec REAL,
    subtitle_type TEXT,           -- manual, auto, external, none
    source_type TEXT,             -- youtube, local
    file_path TEXT,               -- 원본 파일 경로
    audio_path TEXT,              -- 변환된 오디오 경로
    subtitle_path TEXT,           -- 자막 파일 경로
    status TEXT DEFAULT 'pending',
    error_msg TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 세그먼트
CREATE TABLE segments (
    id TEXT PRIMARY KEY,
    media_id TEXT REFERENCES media_items(id),
    segment_index INTEGER,
    start_sec REAL,
    end_sec REAL,
    duration_sec REAL,
    original_text TEXT,
    normalized_text TEXT,
    whisper_text TEXT,
    cer REAL,
    snr_db REAL,
    speech_ratio REAL,
    subtitle_type TEXT,
    audio_path TEXT,
    status TEXT DEFAULT 'pending',
    reject_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 파이프라인 실행 기록
CREATE TABLE pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    status TEXT,                  -- running, completed, stopped, error
    config_snapshot TEXT,         -- 실행 시 설정 JSON
    stats_json TEXT               -- 실행 통계
);

-- 실시간 진행 상황
CREATE TABLE progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES pipeline_runs(id),
    stage TEXT,
    processed INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 구현 순서

### Phase 1: 코어 인프라
1. 프로젝트 구조 생성
2. 설정 관리 (Pydantic)
3. 데이터베이스 모듈
4. 입력 소스 추상화

### Phase 2: 유틸리티
1. 자막 파서 (VTT/SRT)
2. 텍스트 정규화
3. 오디오 유틸리티
4. CER 계산
5. VAD 래퍼

### Phase 3: 파이프라인 스테이지
1. Stage 1: 메타데이터 수집
2. Stage 2: 다운로드/준비
3. Stage 3: CPU 전처리
4. Stage 4: GPU 검증
5. Stage 5: Lhotse 내보내기

### Phase 4: 오케스트레이션
1. 큐 매니저
2. 워커 프로세스
3. 파이프라인 오케스트레이터

### Phase 5: WebUI
1. FastAPI 앱 구조
2. API 라우트
3. WebSocket 핸들러
4. 프론트엔드 템플릿
5. 대시보드 시각화

## 의존성

```
# Core
python >= 3.10
pydantic >= 2.0
pyyaml
aiosqlite
asyncio

# Media Processing
yt-dlp
ffmpeg-python
soundfile
librosa
webvtt-py

# ML/ASR
faster-whisper
silero-vad
torch
jiwer

# Data Export
lhotse

# Web
fastapi
uvicorn
jinja2
python-multipart
websockets

# Utilities
tqdm
aiofiles
httpx
```
