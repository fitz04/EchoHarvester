# EchoHarvester - Agent 작업 지침서

## 프로젝트 개요
한국어 ASR 학습 데이터 파이프라인. YouTube/로컬 미디어에서 음성 데이터를 수집·처리·검증하여 Lhotse Shar 포맷으로 출력.

## 작업 원칙
1. **각 단계별 순차 진행** - 한 단계를 완료한 후 다음 단계로 이동
2. **진행상황 문서화** - 이 파일에 진행상황을 항상 기록하여 세션 중단 시 이어서 작업 가능
3. **에러 기록** - 발생한 에러와 해결 방법을 기록
4. **테스트 후 진행** - 각 단계 완료 후 동작 확인

## 환경 정보
- **OS**: WSL2 (Ubuntu 24.04) on Windows — 기존 macOS에서 이전
- **GPU**: CUDA 지원 가능 (WSL2), `device: "auto"` 사용
- **Python**: 3.12.12 (venv: `./venv/bin/python`)
- **가상환경 활성화**: `source venv/bin/activate`
- **faster-whisper**: CTranslate2 기반 → MPS 미지원, CPU(int8/float32)만 가능
- **네트워크 드라이브**: `Z:` → `/mnt/z/` (drvfs, `\\DESKTOP-I7ITVII\easystore`)
  - 마운트: `wsl.exe -u root -- mount -t drvfs Z: /mnt/z`
  - Shar 데이터: `/mnt/z/data/shar_data/` (~110GB, 16 도메인)

---

## 작업 로드맵

### Phase 1: 환경 설정 및 기본 검증
- [x] 코드 구조 파악
- [x] MacBook GPU 대안 적용 (device: "auto" + CPU 폴백)
- [x] 의존성 설치 (`pip install -e ".[dev]"` → venv/bin/python 3.13)
- [x] ffmpeg 8.0, yt-dlp 2026.02.04 확인
- [x] 기본 import 테스트 (전 모듈 OK)
- [x] config.yaml 검증 테스트 (sources None 버그 수정 포함)

### Phase 2: E2E 파이프라인 테스트 (소규모)
- [x] 테스트용 YouTube 영상 1개로 전체 파이프라인 실행
  - [x] Stage 1: 메타데이터 수집
  - [x] Stage 2: 다운로드
  - [x] Stage 3: CPU 전처리 (FA + VTT dedup 포함)
  - [x] Stage 4: GPU 검증 (Qwen3-ASR-1.7B on MPS)
  - [x] Stage 5: Lhotse 내보내기
- [x] 각 스테이지별 에러 확인 및 수정
- [x] 출력 데이터 검증

### Phase 3: 단위 테스트 작성
- [x] tests/ 디렉토리 구조 생성
- [x] utils/ 모듈 테스트 (text_normalize, cer, subtitle_parser, audio_utils, forced_alignment, retry)
- [x] stages/ 모듈 테스트 (base, metadata, export)
- [x] config 로딩 테스트
- [x] DB 테스트 (CRUD, 마이그레이션, 트랜잭션, 통계)
- [x] sources/ 테스트 (factory, local_file, local_directory, YouTube URL 파싱)
- [x] pipeline orchestrator 테스트

### Phase 4: WebUI 고도화
- [x] 세그먼트 오디오 미리듣기
- [x] 필터링/검색 강화
- [x] 에러 시각화 개선

### Phase 5: 전사 교정 도구 (Transcribe Page)
- [x] DB: APPROVED 상태 추가 + 마이그레이션
- [x] API: 전사 엔드포인트 4개 (media list, segments, approve, reject)
- [x] 라우트: /transcribe 페이지 + 내비게이션 링크
- [x] transcribe.html: wavesurfer.js 파형 에디터 + 3열 텍스트 비교 + 키보드 단축키
- [x] explore.html: "Whisper Text" → "ASR Text" + "Open in Transcribe" 링크
- [x] dashboard.html: approved 상태 색상 매핑
- [x] stage5_export.py: approved 세그먼트 내보내기 포함
- [x] cer.py: docstring 업데이트

### Phase 6: 안정성 및 성능
- [x] 에러 핸들링/리트라이 로직 강화
- [x] 대규모 데이터 처리 최적화
- [x] 로깅 개선

### Phase 7: UI 리디자인 (5 → 3 페이지)
- [x] base.html 내비게이션 3개로 변경 (Home, Process, Review)
- [x] app.py 라우트 변경 + 이전 URL 301 리다이렉트
- [x] home.html: Dashboard + Sources 통합 (인라인 소스 추가, 첫 방문 Welcome, donut 차트)
- [x] process.html: Pipeline 간소화 (수평 스테퍼, 접이식 로그/완료 단계)
- [x] review.html: Explore + Transcribe 마스터-디테일 통합 (WaveSurfer, 필터, 키보드 단축키)
- [x] 이전 템플릿 _archive 폴더로 이동

### Phase 13: k2 + icefall 백엔드 설치
- [x] PyTorch 2.10.0 → 2.9.1 다운그레이드
- [x] k2 1.24.4 pre-built CUDA wheel 설치 (cuda12.8)
- [x] icefall 1.0 editable 설치 (`/mnt/c/work/icefall`)
- [x] k2 CUDA Swoosh 커널 활성화 확인 (`_k2_available = True`)
- [x] 기존 249 tests 전체 PASS
- [x] config.yaml → Zipformer2 CTC로 전환
- [x] Zipformer2 CTC 2 에포크 훈련 검증 (val_loss 2.92)

---

## 현재 진행 상태

### Phase 1 완료 - 환경 설정 및 기본 검증
**시작일**: 2026-02-06
**완료일**: 2026-02-06
**상태**: 완료

#### 완료 항목
- [x] 코드 구조 전체 파악 완료
- [x] GPU 대안 적용: `device: "auto"` 옵션 추가, macOS에서 CPU+int8 자동 폴백
- [x] venv 생성 (Python 3.13.5) + 전체 의존성 설치
- [x] ffmpeg 8.0 + yt-dlp 2026.02.04 확인
- [x] 전체 모듈 import 테스트 통과
- [x] config.yaml 로딩 테스트 통과

### 현재 작업: Phase 2 - E2E 파이프라인 테스트
**시작일**: 2026-02-06
**상태**: 완료 (Stage 1-5 전체 E2E 통과)

#### 테스트 결과 (Run 13, YouTube bynU81TSbDU)

**Stage 3 (CPU 전처리):**
- 입력: VTT 자막 706 cues → dedup 후 354 → merge/split 후 217 세그먼트
- FA(Qwen3-ForcedAligner-0.6B, CPU): 97초 소요
- 결과: 96 cpu_pass / 121 cpu_reject (too_short=105, low_snr=8, low_speech_ratio=8)

**Stage 4 (GPU 검증, Qwen3-ASR-1.7B on MPS):**
- 96개 세그먼트, 6배치, 85.7초 소요
- **16 gpu_pass / 80 gpu_reject** (pass rate: 16.7%)
- CER 분포: 0-0.10=9, 0.10-0.15=7, 0.15-0.30=9, 0.30-0.50=8, 0.50+=63
- CER 전체: avg=0.677, min=0.000, max=1.826

**분석:**
- CER < 0.15 (pass): 자막-ASR 완벽 일치, FA 정확
- CER 0.15-0.30 (near-miss 9개): 내용 맞지만 어미 차이/숫자 표기/끝부분 잘림
- CER 0.50+ (63개): FA 타임스탬프 완전 잘못됨, ASR이 다른 부분 음성 인식

**Stage 5 (Lhotse Shar 내보내기):**
- 16개 gpu_pass 세그먼트 → Lhotse Shar + manifest.jsonl.gz
- 총 103.5초 (1.7분) 오디오
- CER 분포: 0-5%=6, 5-10%=3, 10-15%=7
- 평균 CER: 0.111, 평균 세그먼트 길이: 6.5초
- 출력: `output/shar/` (cuts + recording tar), `output/manifest.jsonl.gz`, `output/stats.json`
- SharWriter API 변경 (lhotse 1.32.2): `fields={"recording": "wav"}` 필수

**향후 개선 방향:**
- 숫자/어미 정규화 추가로 CER 정확도 향상
- 대규모 데이터셋 테스트

### Phase 2.5: FA 정확도 개선
**시작일**: 2026-02-06
**완료일**: 2026-02-06
**상태**: 완료

**문제**: FA 텍스트 매칭에서 구두점(`.`, `?` 등) 불일치로 exact match=0 → 타임스탬프 부정확
**해결**: `_strip_for_matching()`에서 `\W` 패턴으로 구두점+공백 제거, VAD 기반 청킹 적용

#### Before vs After 비교 (YouTube bynU81TSbDU, 15분 영상)

| 지표 | Before | After | 변화 |
|------|--------|-------|------|
| FA exact match | 0/217 | **217/217** | 100% |
| cpu_pass | 96 | **185** | +92.7% |
| cpu_reject | 121 | 32 | -73.6% |
| gpu_pass | 16 | **149** | **+831%** |
| gpu_reject | 80 | 36 | -55.0% |
| Pass rate (전체) | 7.4% | **68.7%** | 9.3x |
| CER avg | 0.677 | **0.121** | -82.1% |
| CER 0-0.10 | 9 | **136** | +1411% |
| CER 0.50+ | 63 | **13** | -79.4% |

**핵심 교훈**: FA 모델(Qwen3-ForcedAligner)은 구두점을 출력하지 않으므로, 매칭 시 원본 텍스트에서도 구두점을 제거해야 정확한 문자열 매칭이 가능

#### 발생한 이슈 및 해결

| 날짜 | 이슈 | 해결 |
|------|------|------|
| 2026-02-06 | macOS에서 CUDA 사용 불가 | device: "auto" 자동 감지 + CPU 폴백 구현 |
| 2026-02-06 | config.yaml에서 sources 주석처리 시 None → ValidationError | field_validator로 None → [] 변환 추가 |
| 2026-02-06 | YouTube VTT 롤링 디스플레이로 텍스트 3중복 | VTT dedup 파서 구현 (706→354 cues) |
| 2026-02-06 | 자막 타임스탬프 부정확 | Qwen3-ForcedAligner-0.6B 강제 정렬 통합 |
| 2026-02-06 | MPS 메모리 부족으로 시스템 크래시 | FA→CPU, 모델 명시적 unload, ASR만 MPS 사용 |
| 2026-02-06 | 로그 부족으로 진행상황 파악 어려움 | Stage 3/4 배치별 상세 로그 + CER 분포 추가 |
| 2026-02-06 | FA 구두점 불일치로 exact match=0 | `_strip_for_matching()`에서 `\W` 패턴으로 제거 |
| 2026-02-06 | Pass 2 per-segment re-alignment 역효과 | Pass 2 제거, VAD 청킹 + 구두점 수정만 적용 |
| 2026-02-06 | 재실행 시 normalized_text가 original_text와 불일치 (stale) | add_segment UPSERT에서 original_text 변경 시 normalized_text/cer NULL 리셋 + Stage4에 일관성 검증 추가 |

### Phase 4: WebUI 고도화
**시작일**: 2026-02-06
**완료일**: 2026-02-06
**상태**: 완료

#### 완료 항목

**Task 1: 세그먼트 오디오 미리듣기**
- `GET /api/dashboard/segments/{segment_id}/audio` 엔드포인트 추가 (FileResponse, DB에서 audio_path 조회)
- explore.html 모달 `<audio>` src를 API 경로로 변경 + `:key` 바인딩으로 리로드 강제
- 테이블에 인라인 재생/정지 버튼 추가 (`playInline()` + `currentAudio` 상태)

**Task 2: 필터링/검색 강화**
- `list_segments` 엔드포인트에 파라미터 추가: cer_min/max, duration_min/max, snr_min/max, text_search, reject_reason, sort_by, sort_order
- `GET /api/dashboard/segments/filter-options` 엔드포인트 추가 (distinct media_ids, reject_reasons 프리픽스, 범위값)
- explore.html 필터 패널 3행 구조로 확장 (Status/Media/Reject/Text, CER/Duration/SNR/Sort, Search/Reset/Limit)
- `resetFilters()` 메서드 추가, `loadFilterOptions()` init 시 호출

**Task 3: 에러 시각화 개선**
- `db.py get_stats()` 확장: reject_reasons (프리픽스 정규화), per_media (pass/reject/avg_cer/pass_duration), cer_distribution_all (7단계 세밀 분포)
- dashboard.html CER Distribution 차트 개선: 전체 validated 세그먼트, 범위별 색상 차등 (green→yellow→orange→red), maxCount 기준 상대 너비
- Charts Row 2 추가: Reject Reasons 수평 바 차트, Per-Media Summary 스택 바 + 통계
- Alpine.js computed: totalRejects, maxCerCount, cerTotal, cerBarColor()

### Phase 5: 전사 교정 도구 (Transcribe Page)
**시작일**: 2026-02-06
**완료일**: 2026-02-06
**상태**: 완료

#### 변경 파일
- `echoharvester/db.py`: APPROVED 상태 enum + `_migrate_schema()` (approved_at, approved_by 컬럼) + get_stats/get_recent_segments에 approved 포함
- `echoharvester/api/routes/dashboard.py`: 전사 API 4개 (GET transcribe/media, GET transcribe/segments/{media_id}, PUT approve, PUT reject)
- `echoharvester/api/app.py`: `/transcribe` 라우트 추가
- `echoharvester/web/templates/base.html`: 내비게이션에 "Transcribe" 링크 추가
- `echoharvester/web/templates/transcribe.html`: 전사 교정 UI (wavesurfer.js v7 파형 에디터, Alpine.js, 3열 텍스트 비교, 키보드 단축키, 세그먼트 스트립, 미디어 사이드바)
- `echoharvester/web/templates/explore.html`: "Whisper Text" → "ASR Text", approved 상태 색상, "Open in Transcribe" 링크
- `echoharvester/web/templates/dashboard.html`: approved 상태 색상 매핑 (파란색)
- `echoharvester/stages/stage5_export.py`: approved 세그먼트도 내보내기에 포함
- `echoharvester/utils/cer.py`: docstring "whisper transcription" → "ASR transcription"

#### 주요 기능
- wavesurfer.js v7 ESM으로 파형 표시 + 재생 제어
- 실시간 CER 계산 (JavaScript Levenshtein distance)
- 키보드 단축키: Space(재생), Tab(다음), Enter(승인), Ctrl+Enter(ASR 사용+승인)
- URL 상태 관리 (media_id + segment_id → 새로고침 시 복귀)
- 자동 전진 모드 (approve 후 다음 세그먼트 자동 이동)

### Phase 3: 단위 테스트 작성
**시작일**: 2026-02-06
**완료일**: 2026-02-06
**상태**: 완료

#### 테스트 결과: 248 passed, 1 skipped, 0 failed

#### 테스트 파일 구조
```
tests/
├── conftest.py              # 공유 fixtures (DB, config, 샘플 오디오/자막)
├── test_text_normalize.py   # 텍스트 정규화 (38 tests)
├── test_cer.py              # CER 계산 (18 tests)
├── test_subtitle_parser.py  # 자막 파싱 (23 tests)
├── test_audio_utils.py      # 오디오 유틸리티 (15 tests)
├── test_forced_alignment.py # FA 유틸리티 (13 tests)
├── test_config.py           # 설정 관리 (16 tests)
├── test_db.py               # DB CRUD/트랜잭션/통계 (27 tests)
├── test_sources.py          # 소스 클래스 (17 tests)
├── test_stages.py           # 스테이지 기본 클래스 (7 tests)
├── test_pipeline.py         # 파이프라인 오케스트레이터 (11 tests)
└── test_retry.py            # 리트라이 유틸리티 (8 tests)
```

#### 커버리지 (주요 모듈)
- `utils/text_normalize.py`: 99%
- `utils/cer.py`: 85%
- `utils/subtitle_parser.py`: 80%
- `utils/retry.py`: 98%
- `config.py`: 84%
- `db.py`: 93%
- `pipeline/orchestrator.py`: 82%
- `sources/base.py`: 96%
- `stages/base.py`: 94%

#### 발견 및 수정된 버그
- `text_normalize.py`: 유니코드 따옴표(`\u201c\u201d`) 정규화 regex에 실제 유니코드 포인트 대신 ASCII `"` 사용 → 수정 완료

### Phase 6: 안정성 및 성능
**시작일**: 2026-02-06
**완료일**: 2026-02-06
**상태**: 완료

#### 변경 사항

**에러 핸들링/리트라이**
- `echoharvester/utils/retry.py`: 범용 `retry`/`async_retry` 데코레이터 추가 (지수 백오프, 콜백)
- `echoharvester/utils/audio_utils.py`: `convert_to_wav()`에 `max_retries` 파라미터 + 재시도 로직
- `echoharvester/stages/stage4_validate.py`: 모델 초기화 3회 재시도 + GC 정리
- `echoharvester/stages/stage2_download.py`: 에러 로그에 예외 타입 포함 + exc_info 추가
- `echoharvester/stages/stage3_preprocess.py`: 에러 로그에 예외 타입 포함 + exc_info 추가

**로깅 개선**
- `echoharvester/stages/base.py`: 스테이지 시작/완료 타이밍 로그 + elapsed_sec 통계 반환
- `echoharvester/db.py`: DB 연결/해제 로그 추가
- `echoharvester/pipeline/orchestrator.py`: 에러 로그에 예외 타입 포함

**대규모 데이터 최적화**
- `echoharvester/db.py`: WAL 저널 모드 + busy_timeout=5000ms (동시 읽기 성능 향상)
- `echoharvester/db.py`: `bulk_update_segments()` 500건 단위 청크 분할 (SQLite 변수 제한 방지)

### Phase 7: UI 리디자인 (5 → 3 페이지)
**시작일**: 2026-02-07
**완료일**: 2026-02-07
**상태**: 완료

#### 변경 파일
- `echoharvester/web/templates/base.html`: Nav 5개 → 3개 (Home, Process, Review), 파이프라인 상태 링크 /process로 변경
- `echoharvester/api/app.py`: 라우트 변경 (/ → home.html, /process → process.html, /review → review.html) + 이전 URL 301 리다이렉트 (query params 보존)
- **새로 생성** `echoharvester/web/templates/home.html`: Dashboard + Sources 통합
  - 인라인 소스 추가 (URL 자동 타입 감지), 첫 방문 Welcome hero, Quick Stats 4카드
  - 소스 테이블 + CSS donut 차트 + 최근 활동 피드 + Process All 버튼
- **새로 생성** `echoharvester/web/templates/process.html`: Pipeline 간소화
  - 수평 스테퍼 (5단계 원/라인), 활성 스테이지 상세 진행바, 완료 단계 아코디언
  - Advanced 접힘 영역에 개별 스테이지 Run 버튼, 접이식 로그 패널
  - 완료 배너 + "Review Results" CTA 버튼
- **새로 생성** `echoharvester/web/templates/review.html`: Explore + Transcribe 마스터-디테일 통합
  - 좌측 세그먼트 리스트 (40%) + 우측 상세 패널 (60%)
  - 퀵필터 칩 (All/Needs Review/Approved/Rejected), 간소화된 필터바 + "More" 접힘
  - WaveSurfer.js 파형, 3열 텍스트 비교, 접이식 메타데이터, 액션 버튼
  - 키보드 단축키 (Space/Tab/Enter/Ctrl+Enter), auto-advance, URL 딥링크
- 이전 템플릿 5개 → `_archive/` 폴더로 이동

### Phase 9: Training Web UI
- [x] Training 모듈에 progress_callback + stop 메커니즘 추가
- [x] Backend API 라우트 (12 엔드포인트)
- [x] WebSocket `/ws/training` 엔드포인트
- [x] Navigation + 앱 라우트 등록
- [x] Training 페이지 3탭 (Setup, Monitor, Results)
- [x] Canvas 기반 Loss/LR 차트
- [x] Character-level diff 하이라이트
- [x] WaveSurfer.js 디코드 샘플 오디오 재생
- [x] 모델 Export (ONNX/TorchScript)

### Phase 10: 네트워크 드라이브 + 테스트 Shar 데이터
- [x] WSL2에서 네트워크 드라이브 마운트 (Z: → /mnt/z/)
- [x] config.yaml에 테스트 shar_sources 3개 추가 (temp, radio, medical; ~412MB)
- [x] .bak 파일 자동 무시 확인
- [ ] Training Web UI에서 Prepare Data + Training 실행 검증

### Phase 8: Icefall 한국어 ASR 훈련 모듈
**시작일**: 2026-02-08
**완료일**: 2026-02-08
**상태**: 완료 (Phase 1~4 구현 + 검증)

#### 개요
EchoHarvester가 생성하는 Lhotse Shar 데이터를 활용하여 icefall 스타일 Conformer CTC 한국어 ASR 엔진을 훈련하는 기능 추가. icefall을 import하지 않고 standalone 구현 (의존성 최소화).

#### 새 파일
```
echoharvester/training/
├── __init__.py          # 모듈 exports
├── config.py            # TrainingConfig (Pydantic: split, tokenizer, model, training_params)
├── data_prep.py         # DataPreparer: Shar 로드→오디오 디스크 저장→media_id 기준 stratified split
├── tokenizer.py         # CharTokenizer: 한글 음절 character-level tokens.txt (432 토큰)
├── model.py             # ConformerCtcModel: Conv2dSubsampling→PE→ConformerEncoder×12→CTC
├── dataset.py           # AsrDataset + create_dataloader (Lhotse SimpleCutSampler + on-the-fly Fbank)
├── utils.py             # resolve_device, set_seed, NoamScheduler, compute_cer
├── trainer.py           # Trainer: 에포크 루프, 체크포인트, 검증, TensorBoard
├── decode.py            # Decoder: CTC greedy decode + CER 평가
└── export.py            # ModelExporter: ONNX/TorchScript 내보내기
```

#### 수정된 기존 파일
- `echoharvester/config.py`: `training: dict | None` 필드 + `get_training_config()` 메서드
- `echoharvester/main.py`: `train` 서브커맨드 그룹 (prepare, run, status, decode, export)
- `config.yaml`: `training:` 섹션 추가

#### CLI 명령어
```
echoharvester train prepare     # 데이터 분할 + 토크나이저 생성
echoharvester train run         # 모델 훈련 (--epochs, --resume 지원)
echoharvester train status      # 훈련 상태 확인
echoharvester train decode      # CTC greedy decode + CER 평가 (--checkpoint)
echoharvester train export      # ONNX/TorchScript 내보내기 (--checkpoint, --format)
```

#### 검증 결과 (E2E 테스트, 120 cuts / 743초)
- `train prepare`: 118 train / 1 val / 1 test, 432 토큰 (411 한글 음절 + 18 기타 + 3 특수)
- `train run --epochs 2`: loss 21.54→15.95, val_loss 22.47→10.59, 32.9M params, ~3초/에포크 (CUDA)
- `train run --epochs 4 --resume`: loss 8.01→6.13, val_loss 6.32→6.06 (정상 수렴)
- `train decode`: CER=1.0 (2에포크에서는 미학습 상태, 정상 동작 확인)
- `train export --format torchscript`: 135.9 MB 모델 정상 출력
- 기존 249 tests 전체 PASS (regression 없음)

#### 모델 아키텍처 (Conformer CTC, 32.9M params)
```
Fbank (batch, time, 80)
  → Conv2dSubsampling (time//4, 256)
  → PositionalEncoding
  → ConformerEncoderLayer × 12 (Macaron-net: FF→MHSA→Conv→FF)
  → Linear(256, 432) → log_softmax → CTC Loss
```

#### 핵심 구현 결정
- Shar 데이터 로드 시 in-memory 오디오를 디스크에 WAV로 저장 (JSONL lazy loading 호환)
- `SimpleCutSampler` 사용 (`DynamicBucketingSampler`는 소규모 데이터에서 drop_last 문제)
- `shard_origin` PosixPath → str 변환, `dataloading_info` 제거 (Lhotse 직렬화 호환)
- ONNX export는 `onnxscript` 패키지 필요 (optional dependency)
- TorchScript export에서 `check_trace=False` (dropout으로 인한 graph 차이 무시)

### Phase 8.5: Zipformer2 CTC 모델 추가
**시작일**: 2026-02-08
**완료일**: 2026-02-08
**상태**: 완료

#### 개요
기존 Conformer CTC 모델에 더하여 icefall의 Zipformer2 아키텍처를 추가. icefall/k2를 pip 설치할 필요 없이 Apache 2.0 소스 3개를 vendor하여 독립 실행 가능.

#### 변경 사항

**새 파일 (vendored from k2-fsa/icefall)**
```
echoharvester/training/zipformer/
├── __init__.py       # Zipformer2, Conv2dSubsampling exports
├── scaling.py        # BiasNorm, SwooshR/L, ActivationBalancer, ScaledLinear 등
├── subsampling.py    # Conv2dSubsampling (Zipformer용)
├── zipformer.py      # Zipformer2 encoder
└── LICENSE           # Apache 2.0 (k2-fsa/icefall)
```

**수정 파일**
- `echoharvester/training/config.py`: `ModelConfig` → `ConformerModelConfig` + `ZipformerModelConfig` discriminated union
- `echoharvester/training/model.py`: `ZipformerCtcModel` 래퍼 클래스 + `create_model()` 팩토리
- `echoharvester/training/trainer.py`: `ConformerCtcModel(...)` → `create_model(...)` + d_model 분기
- `echoharvester/training/decode.py`: `ConformerCtcModel(...)` → `create_model(...)`
- `echoharvester/training/export.py`: `ConformerCtcModel(...)` → `create_model(...)`
- `config.yaml`: Zipformer 설정 예시 주석 추가

#### Vendor 수정 사항
- `from scaling import ...` → `from echoharvester.training.zipformer.scaling import ...`
- `from encoder_interface import EncoderInterface` → 제거 (nn.Module 직접 상속)
- `from icefall.utils import torch_autocast` → 로컬 shim 함수
- `import k2` → optional import (k2 없으면 pure-torch SwooshL/R fallback)
- `torch.cuda.amp.custom_fwd/bwd` → `torch.amp.custom_fwd/bwd` (FutureWarning 수정)

#### 검증 결과
- Conformer CTC: 32.9M params, forward pass OK
- Zipformer2 CTC: 63.6M params (default config), forward pass OK
- Config discriminated union: YAML 파싱 OK
- 기존 tests 전체 PASS (regression 없음)

### Phase 9: Training Web UI
**시작일**: 2026-02-08
**완료일**: 2026-02-08
**상태**: 완료

#### 개요
Training 모듈(Conformer CTC / Zipformer2 CTC)을 웹에서 제어할 수 있는 Training Web UI 구현.
기존 WebUI 스택(FastAPI + Jinja2 + Alpine.js + Tailwind CSS + WebSocket) 그대로 사용.

#### 변경 사항

**새 파일**
- `echoharvester/api/routes/training.py`: Training REST API (12 엔드포인트)
  - GET /api/training/config, data-stats, checkpoints, status, decode-results
  - POST /api/training/prepare, start, stop, decode, export
  - GET /api/training/decode-results/audio/{idx}
- `echoharvester/web/templates/training.html`: Training 페이지 (3탭 UI)

**수정 파일**
- `echoharvester/api/app.py`: `/training` HTML 라우트 + training 라우터 등록 + nav 폴백 추가
- `echoharvester/api/routes/__init__.py`: `training` import 추가
- `echoharvester/api/routes/websocket.py`: `training_manager` + `/ws/training` 엔드포인트 추가
- `echoharvester/web/templates/base.html`: nav에 "Training" 링크 + `nav_training` block
- `echoharvester/training/trainer.py`: `progress_callback`, `_stop_requested`, `_emit()`, stop 체크
- `echoharvester/training/data_prep.py`: `progress_callback` + 단계별 이벤트 발행
- `echoharvester/training/decode.py`: `progress_callback`, `max_samples` 파라미터, 진행 이벤트
- `echoharvester/training/export.py`: `progress_callback` + start/complete 이벤트

#### 3탭 구성
1. **Setup**: 데이터 준비 + 모델 설정 + 체크포인트 목록 + Start Training
2. **Monitor**: 실시간 loss curve (Canvas), 진행률 바, 로그 패널, Stop 버튼, WebSocket
3. **Results**: 디코딩 컨트롤, CER 분포, 오류 샘플 master-detail, WaveSurfer 오디오, char-level diff, Export

#### WebSocket 이벤트 체계
- Thread→Async 브릿지: `asyncio.run_coroutine_threadsafe()` 사용
- 이벤트: prepare_*, train_start/epoch_start/batch_update/epoch_complete/train_complete/train_stopped, decode_*, export_*
- 연결 시 현재 상태 전송 (브라우저 새로고침 대응)

#### 검증
- 기존 249 tests 전체 PASS (regression 없음)
- 모든 import 체인 정상
- Training 모듈 callback=None 시 기존 CLI 동작 그대로 유지
- 앱 생성 + 라우트 등록 정상 (총 44 routes)

### Phase 10: 네트워크 드라이브 마운트 + 테스트 Shar 데이터
**시작일**: 2026-02-08
**완료일**: 2026-02-08
**상태**: 완료

#### 개요
WSL2에서 네트워크 드라이브 `\\DESKTOP-I7ITVII\easystore`를 마운트하여 대규모 한국어 ASR Shar 데이터(~110GB, 16 도메인)에 접근. 테스트용으로 소형 3개 디렉토리 선택.

#### 마운트 설정
- Windows에서 `net use Z: \\DESKTOP-I7ITVII\easystore /persistent:yes`
- WSL2에서 `wsl.exe -u root -- mount -t drvfs Z: /mnt/z`
- 경로: `/mnt/z/data/shar_data/`

#### 테스트 Shar 소스 (config.yaml)

| 디렉토리 | Shards | 크기 | .bak 파일 |
|----------|--------|------|-----------|
| `/mnt/z/data/shar_data/temp` | 5 | 79MB | 4 (무시됨) |
| `/mnt/z/data/shar_data/radio` | 9 | 159MB | 9 (무시됨) |
| `/mnt/z/data/shar_data/medical` | 4 | 174MB | 4 (무시됨) |
| **합계** | **18** | **~412MB** | **17** |

#### .bak 파일 처리
- `Path.glob("cuts.*.jsonl.gz")`는 `.bak`을 매칭하지 않지만 (Python pathlib)
- **Lhotse `CutSet.from_shar(in_dir=...)`는 `extension_contains(".jsonl")`를 사용하여 `.bak` 파일도 매칭** → 버그 발생
- **수정**: `data_prep.py`의 `_load_shar_sources()`에서 `in_dir` 대신 `fields` 파라미터로 명시적 파일 목록 전달, `.bak` 파일 제외
- 결과: temp 4,660 cuts (5.5h), radio 8,677 cuts (11.2h), medical 3,937 cuts (14.0h) → **총 17,274 cuts (30.7h)**

#### 다음 단계
- Training Web UI에서 Prepare Data → Start Training 실행
- Monitor 탭에서 loss 수렴 확인
- 향후 필요 시 대규모 데이터 랜덤 샘플링 유틸리티 구현

### Phase 11: Pretrained Model Fine-tuning 지원
**시작일**: 2026-02-08
**상태**: 완료

#### 개요
기존 스크래치 훈련만 지원하던 Training 모듈에 프리트레인 모델 파인튜닝 기능 추가.
동일 아키텍처 내 key-by-key 가중치 로딩, 인코더 프리징, epoch 기반 warmup 지원.

#### 변경 파일 (5개)

**`echoharvester/training/config.py`**
- `TrainingParamsConfig`에 `warmup_epochs: float = 0` 필드 추가
- `warmup_epochs > 0`이면 `warm_step` 자동 계산 (에폭 비율 × steps_per_epoch)

**`echoharvester/training/trainer.py`**
- `_load_pretrained(model, path)`: key-by-key shape 매칭, 불일치 시 skip + 로그
- `_freeze_encoder(model)`: output_proj 외 전체 파라미터 freeze
- `_unfreeze_all(model)`: 전체 파라미터 unfreeze
- `train()` 시그니처 확장: `pretrained_checkpoint`, `freeze_encoder_epochs` 추가
- 데이터로더 생성 후 `warmup_epochs → warm_step` 변환 (NoamScheduler 생성 전)
- 에폭 루프에 freeze/unfreeze 로직 (start_epoch 기준)
- 새 이벤트: `pretrained_loaded`, `encoder_frozen`, `encoder_unfrozen`

**`echoharvester/main.py`**
- `train run` 서브파서에 `--pretrained`, `--freeze-encoder-epochs` 인자 추가
- `trainer.train()` 호출 시 전달

**`echoharvester/api/routes/training.py`**
- POST `/training/start` body에서 `pretrained_checkpoint`, `freeze_encoder_epochs` 수용
- `lr_factor`, `warmup_epochs`, `warm_step` 오버라이드도 body에서 수용
- `trainer.train()` 호출 시 전달

**`echoharvester/web/templates/training.html`**
- 상태 변수: `pretrainedCheckpoint`, `freezeEncoderEpochs`
- Training Parameters에 `Warmup Epochs` 필드 추가 (>0이면 Warmup Steps 비활성)
- Device/Resume 아래에 Fine-tuning 행: Pretrained Model 드롭다운 + Freeze Encoder 입력
  - Resume 선택 시 Fine-tuning 행 숨김 (상호배타)
  - Pretrained 미선택 시 Freeze 비활성
- `onPretrainedChange()`: 파인튜닝 선택 시 lr_factor=0.5, warmup_epochs=0.5, freeze=3 프리셋
- `startTraining()`: body에 pretrained/freeze/lr_factor/warmup_epochs/warm_step 전송
- WS 이벤트 핸들러: `pretrained_loaded`, `encoder_frozen`, `encoder_unfrozen` → 로그 출력

### Phase 12: Training UI 프리셋 자동 적용
**시작일**: 2026-02-09
**상태**: 완료

#### 변경 파일 (1개)
- `echoharvester/web/templates/training.html`

#### 변경 내용
- `MODEL_PRESETS` 상수 추가 (Conformer CTC / Zipformer CTC 기본값)
- 초기 `config.model`을 프리셋 기반으로 변경 (빈 문자열 제거)
- Model Type 라디오 버튼에 `@change="onModelTypeChange()"` 추가
- `onModelTypeChange()`: 타입 전환 시 해당 프리셋으로 전체 교체
- `loadConfig()`에서 API 응답 머지 시 프리셋 기반 폴백 적용
- Zipformer UI 필드 확장: encoder_dim + num_encoder_layers + dropout 기본 표시, Advanced 접힘 영역에 num_heads, feedforward_dim, cnn_module_kernel, downsampling_factor 추가

### Phase 12.5: Training 런타임 버그 수정
**시작일**: 2026-02-09
**상태**: 완료

#### 수정 사항

**버그 1: PositionalEncoding 시퀀스 길이 초과 크래시**
- **증상**: 긴 오디오(~145초+)에서 `RuntimeError: The size of tensor a (14523) must match the size of tensor b (10000)`
- **원인**: `PositionalEncoding.max_len=10000` 고정, Conv2dSubsampling 후에도 10,000 프레임 초과하는 시퀀스 존재
- **수정**: `model.py` — `_extend_pe()` 메서드 추가, `forward()` 시 시퀀스 길이가 버퍼 초과하면 동적으로 PE 재계산
- **결과**: 어떤 길이의 오디오도 크래시 없이 처리

**버그 2: SimpleCutSampler `len()` 미지원 TypeError**
- **증상**: `warmup_epochs > 0` + `stats.json` 없을 때 `TypeError: object of type 'SimpleCutSampler' has no len()`
- **원인**: `trainer.py` 폴백 분기에서 `len(train_dl)` 호출 시 Lhotse `SimpleCutSampler`가 `__len__` 미구현
- **수정**: `trainer.py` — `try: len(train_dl)` / `except TypeError:` 로 sampler 반복 카운트 폴백
- **결과**: stats.json 유무와 관계없이 정상 동작

#### 변경 파일 (2개)
- `echoharvester/training/model.py`: `PositionalEncoding._extend_pe()` 추가 + `forward()` 수정
- `echoharvester/training/trainer.py`: `warm_step` 계산 폴백 로직 try/except 래핑

#### 검증
- 기존 249 tests 전체 PASS (regression 없음)

### Phase 13: k2 + icefall 백엔드 설치
**시작일**: 2026-02-09
**상태**: 완료

#### 개요
기존 Zipformer2 코드는 icefall에서 vendor한 pure-PyTorch 구현 사용.
k2 설치로 Swoosh 활성 함수 CUDA 커널 활성화 → 훈련 5-20% 속도 향상 기대.
icefall editable 설치로 향후 N-best/Lattice 디코딩, HLG 그래프 디코딩 가능.

#### 설치 내역

| 패키지 | 변경 전 | 변경 후 |
|--------|---------|---------|
| PyTorch | 2.10.0+cu126 | **2.9.1+cu128** |
| torchaudio | 2.10.0+cu126 | **2.9.1** |
| k2 | (없음) | **1.24.4.dev20251118+cuda12.8.torch2.9.1** |
| icefall | (없음) | **1.0** (editable, `/mnt/c/work/icefall`) |

**PyTorch 다운그레이드 사유**: k2 pre-built wheel이 2.10.0에는 미제공, 2.9.1에는 제공됨.
코드베이스에 PyTorch 2.10 전용 API 사용 없음 → 다운그레이드 안전.

**CUDA 버전 주의**: PyTorch 2.9.1은 cu12.8 번들. k2 wheel도 반드시 `cuda12.8` 태그 사용 필요.
(cuda12.9 wheel 설치 시 `ImportError: k2 was built using CUDA 12.9 But you are using CUDA 12.8` 발생)

#### k2 CUDA 커널 활성화 확인
- `scaling.py`의 `_k2_available = True` 자동 설정
- `SwooshL`, `SwooshR` CUDA forward+backward pass 정상 동작
- 별도 코드 수정 없음 (기존 `try: import k2` 패턴으로 자동 활성화)

#### icefall 추가 의존성 (자동 설치됨)
- kaldifst, kaldilm, kaldialign, kaldi-decoder, sentencepiece
- tensorboard, onnx, onnxruntime, onnxoptimizer, onnxsim
- num2words, pypinyin, pycantonese, typeguard, dill

#### 검증
- `import k2` → OK, `k2.with_cuda = True`
- `k2.swoosh_l(torch.randn(10, device='cuda'))` → CUDA 커널 정상 동작
- `import icefall` → OK (`/mnt/c/work/icefall/icefall/__init__.py`)
- 기존 249 tests 전체 PASS (regression 없음)

#### k2 Swoosh CUDA 커널 벤치마크

| 구분 | k2 CUDA | pure-torch | 속도 향상 |
|------|---------|------------|-----------|
| Forward only | ~34ms | ~35ms | ~1.0x (차이 없음) |
| **Forward+Backward** | **244ms** | **312ms** | **~1.28x** |

- Forward는 차이 없으나 **backward pass에서 ~28% 빠름** (fused kernel으로 메모리 접근 감소)
- Swoosh가 모델 전체 연산의 일부이므로 실제 훈련 속도 향상은 **5~10%** 수준

#### Zipformer2 CTC 훈련 검증 (k2 활성 상태)

| 항목 | 값 |
|------|-----|
| 모델 | Zipformer2 CTC, **64.2M params** |
| 데이터 | 17,274 cuts (~30.7h), 3 도메인 (temp/radio/medical) |
| Epoch 1 | train_loss: 8.37, val_loss: 4.29, 601.8초 |
| Epoch 2 | train_loss: 3.47, val_loss: 2.92, 593.6초 |
| 50 steps 간격 | ~25.5초 (안정적) |
| 체크포인트 | `exp/epoch-1.pt`, `exp/epoch-2.pt`, `exp/best.pt` |

#### config.yaml 변경
- `training.model.type`: `conformer_ctc` → **`zipformer_ctc`** 로 전환
- Conformer 설정은 주석 처리하여 보존
- 현재 활성 모델: Zipformer2 CTC (6스택, 192~512 dim)

---

## 향후 계획

### Zipformer2 본격 훈련 (50 에포크)
- 현재 2 에포크만 돌림 (val_loss 2.92), 50 에포크까지 `--resume` 사용
- `python -c "from echoharvester.main import main; import sys; sys.argv = ['e', 'train', 'run', '--epochs', '50', '--resume']; main()"`
- 또는 Training Web UI에서 Resume 체크포인트 선택 후 Start

### Icefall 허브 프리트레인 모델 다운로드
- Icefall에서 제공하는 공개 한국어 모델을 UI에서 직접 다운로드
- 다운로드한 모델의 아키텍처 파라미터 자동 반영 (config 역매핑)
- 현재는 `.pt` 파일을 `./exp/` 폴더에 수동 복사하면 Pretrained Model 목록에 표시됨

### k2 Lattice 디코딩 통합
- k2 설치 완료로 N-best / Lattice 디코딩 구현 가능
- HLG 그래프 디코딩, LM 통합 (향후)

---

## 기술 노트

### MacBook에서 GPU/모델 관리
- `faster-whisper`는 CTranslate2 기반으로 **MPS(Apple GPU) 미지원**
- macOS에서는 `device: "cpu"` + `compute_type: "int8"` 조합이 최적
- `device: "auto"` 설정 시 자동으로 CUDA > CPU 순으로 폴백
- Silero VAD는 PyTorch 기반이지만 CPU로도 충분히 빠름
- **Qwen3-ASR-1.7B**: MPS bfloat16 동작 확인 (M3 Pro 18GB)
- **Qwen3-ForcedAligner-0.6B**: CPU float32 사용 (MPS 메모리 충돌 방지)
- **중요**: FA와 ASR을 동시에 MPS 로드하면 메모리 부족으로 시스템 크래시 발생
  → FA를 CPU로 강제하고 완료 후 명시적 unload 필수

### 주요 설정 파일
- `config.yaml`: 파이프라인 설정 (소스, 필터링, GPU 등)
- `pyproject.toml`: 패키지 의존성
- `echoharvester/config.py`: Pydantic 설정 모델

### WSL2 네트워크 드라이브 마운트
- Windows에서 `net use Z: \\DESKTOP-I7ITVII\easystore /persistent:yes`
- WSL2 부팅 후 매번 마운트 필요: `wsl.exe -u root -- bash -c "mkdir -p /mnt/z && mount -t drvfs Z: /mnt/z"`
- WSL2는 9p 프로토콜 사용, drvfs 마운트된 네트워크 드라이브는 읽기 성능이 로컬보다 느림
- `/etc/fstab` 영구 마운트 가능: `Z: /mnt/z drvfs defaults 0 0`

### k2 + icefall 설치 재현 (새 환경에서)
```bash
source venv/bin/activate
pip install torch==2.9.1 torchaudio==2.9.1
# CUDA 버전 확인: python -c "import torch; print(torch.version.cuda)"
# cu12.8이면 cuda12.8 태그, cu12.6이면 cuda12.6 태그 사용
pip install k2==1.24.4.dev20251118+cuda12.8.torch2.9.1 -f https://k2-fsa.github.io/k2/cuda.html
cd /mnt/c/work/icefall && pip install -e .
# 검증: python -c "import k2; print(k2.with_cuda)"
```
- k2 wheel 목록: https://k2-fsa.github.io/k2/cuda.html
- **주의**: k2 wheel의 CUDA 버전이 PyTorch 번들 CUDA 버전과 정확히 일치해야 함

### 파이프라인 상태 관리
- SQLite DB에 모든 상태 저장 → 중단 후 재개 가능
- `auto_resume: true` 설정으로 자동 재개 지원
