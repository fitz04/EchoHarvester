# EchoHarvester - Claude Code 프로젝트 지침

## Project Context
한국어 ASR 학습 데이터 파이프라인 (YouTube/로컬 미디어 → Lhotse Shar 포맷).
각 Claude Code 세션은 독립적이며 이전 세션의 컨텍스트를 공유하지 않습니다.
작업 시작 전 반드시 `agent.md`를 읽어 현재 진행 상태를 파악한 후 변경하세요.

## Environment
- macOS (Apple Silicon, M3 Pro 18GB) — CUDA/GPU 지원 없음
- Python 3.13.5 (venv: `./venv/bin/python`)
- 가상환경 활성화: `source venv/bin/activate`
- GPU/CUDA 작업 시도 금지 — 항상 CPU 호환 코드 경로 사용
- faster-whisper: CTranslate2 기반, MPS 미지원 → CPU + int8/float32
- Qwen3-ASR-1.7B: MPS bfloat16 사용 가능 (검증 스테이지)
- Qwen3-ForcedAligner-0.6B: CPU float32 전용 (MPS 메모리 충돌 방지)
- FA와 ASR 동시 MPS 로드 금지 → FA는 CPU 강제 + 완료 후 명시적 unload

## Documentation
- 코드 변경 후 관련 문서(`agent.md` 등)를 반드시 동기화
- 문서 업데이트 시 관련 파일 전체를 한 세션에서 완료 — 부분 업데이트 금지
- `agent.md`: 작업 로드맵 + 진행 상태 + 이슈 기록 (항상 최신 유지)

## Project Structure
```
echoharvester/
├── config.py          # Pydantic 설정 모델
├── db.py              # SQLite 상태 관리
├── sources/           # YouTube 등 소스 처리
├── stages/
│   ├── stage1_metadata.py
│   ├── stage2_download.py
│   ├── stage3_preprocess.py  # CPU 전처리 (FA + VTT dedup)
│   ├── stage4_validate.py    # GPU/Whisper 검증
│   └── stage5_export.py      # Lhotse Shar 내보내기
├── utils/
│   ├── forced_alignment.py
│   ├── subtitle_parser.py
│   └── ...
├── api/
│   ├── app.py         # FastAPI 앱 (3페이지: /, /process, /review)
│   └── routes/
├── web/templates/
│   ├── base.html      # 기본 레이아웃 (Home, Process, Review 내비)
│   ├── home.html      # 소스 관리 + 대시보드 통합
│   ├── process.html   # 파이프라인 제어 (수평 스테퍼)
│   └── review.html    # 탐색 + 전사 교정 (마스터-디테일)
config.yaml            # 파이프라인 설정
agent.md               # 작업 지침서 + 진행 상태
```
