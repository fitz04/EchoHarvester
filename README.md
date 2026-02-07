# EchoHarvester

Korean ASR (Automatic Speech Recognition) training data pipeline.

Collects Korean subtitles and audio from YouTube and local media files, then processes them through a 5-stage pipeline to produce high-quality ASR training data in [Lhotse Shar](https://github.com/lhotse-speech/lhotse) format.

## Features

- **Multiple Input Sources**
  - YouTube channels, playlists, and individual videos
  - Local video/audio files
  - Batch directory processing

- **5-Stage Pipeline**
  1. **Metadata** — Collect video metadata and subtitle info
  2. **Download** — Download media and extract audio
  3. **Preprocess** — Forced alignment, subtitle dedup, normalization, SNR/VAD filtering
  4. **Validate** — ASR re-transcription with Qwen3-ASR, CER-based filtering
  5. **Export** — Lhotse Shar archive output

- **Forced Alignment**
  - Qwen3-ForcedAligner-0.6B for precise speech-text timestamp alignment
  - VAD-based chunking with Silero VAD for natural boundary detection
  - Punctuation-normalized matching between FA output and source subtitles

- **Transcription Correction**
  - WaveSurfer.js waveform editor with playback controls
  - 3-column text comparison (Original / Corrected / ASR)
  - Live CER calculation as you edit
  - Keyboard shortcuts for efficient review workflow
  - Auto-advance mode after approval

- **3-Page WebUI**
  - **Home** — Source management + overview dashboard
  - **Process** — Pipeline control with live progress tracking
  - **Review** — Segment exploration + transcription correction (master-detail layout)

## Installation

```bash
# Clone the repository
git clone https://github.com/fitz04/EchoHarvester.git
cd EchoHarvester

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Prerequisites

- **Python** >= 3.10
- **ffmpeg** — required for audio processing
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg
  # macOS
  brew install ffmpeg
  ```

- **CUDA** (optional) — for GPU-accelerated validation
  ```bash
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- **macOS (Apple Silicon)** — works without CUDA
  - `faster-whisper` uses CTranslate2 (no MPS support) — auto-falls back to CPU with int8 quantization
  - Set `device: "auto"` in config for automatic detection

## Usage

### 1. Configuration

Edit `config.yaml` to set up your input sources:

```yaml
sources:
  - type: youtube_channel
    url: "https://www.youtube.com/@channel_name"
    label: "News Channel"

  - type: local_directory
    path: "/data/videos"
    pattern: "*.mp4"
    label: "Local Videos"

filters:
  cer_threshold_manual: 0.15    # CER threshold for manual subtitles
  cer_threshold_auto: 0.10      # CER threshold for auto subtitles

validation:
  model: "seastar105/whisper-medium-komixv2"
  device: "auto"                # auto-detect (CUDA > CPU)
```

### 2. CLI

```bash
# Run the full pipeline
echoharvester run --config config.yaml

# Run a specific stage
echoharvester run --config config.yaml --stage preprocess

# Check pipeline status
echoharvester status --config config.yaml

# View errors
echoharvester errors --config config.yaml

# Generate statistics report
echoharvester report --config config.yaml
```

### 3. WebUI

```bash
# Start the web server
echoharvester server --config config.yaml

# Open in browser
# http://127.0.0.1:8000
```

#### Pages

| Page | URL | Description |
|------|-----|-------------|
| **Home** | `/` | Add sources, view stats, start processing |
| **Process** | `/process` | Monitor pipeline progress, view logs |
| **Review** | `/review` | Browse segments, edit transcriptions, approve/reject |

#### Keyboard Shortcuts (Review page)

| Key | Action |
|-----|--------|
| `Space` | Play / Pause audio |
| `Left/Right` | Seek 2 seconds |
| `Ctrl+Left/Right` | Seek 5 seconds |
| `Tab` | Next segment |
| `Shift+Tab` | Previous segment |
| `Enter` | Approve current segment |
| `Ctrl+Enter` | Use ASR text + Approve |
| `?` | Toggle shortcut help |

## Project Structure

```
echoharvester/
├── config.py                # Pydantic configuration models
├── db.py                    # SQLite state management
├── main.py                  # CLI entry point
├── sources/                 # Input source implementations
│   ├── youtube.py
│   ├── local_file.py
│   └── local_directory.py
├── stages/                  # Pipeline stages
│   ├── stage1_metadata.py
│   ├── stage2_download.py
│   ├── stage3_preprocess.py # CPU preprocessing (FA + VTT dedup)
│   ├── stage4_validate.py   # ASR validation (Qwen3-ASR)
│   └── stage5_export.py     # Lhotse Shar export
├── utils/                   # Utility modules
│   ├── forced_alignment.py  # Qwen3-ForcedAligner integration
│   ├── subtitle_parser.py   # VTT/SRT parser with dedup
│   ├── text_normalize.py    # Korean text normalization
│   ├── audio_utils.py       # Audio conversion utilities
│   ├── cer.py               # Character Error Rate calculation
│   └── vad.py               # Voice Activity Detection
├── pipeline/                # Pipeline orchestrator
│   └── orchestrator.py
├── api/                     # FastAPI backend
│   ├── app.py
│   └── routes/
│       ├── sources.py
│       ├── pipeline.py
│       ├── dashboard.py
│       └── websocket.py
└── web/                     # WebUI
    ├── templates/
    │   ├── base.html        # Base layout (3-page nav)
    │   ├── home.html        # Source management + dashboard
    │   ├── process.html     # Pipeline control
    │   └── review.html      # Segment review + transcription
    └── static/
```

## Output Format

Processed data is exported in Lhotse Shar format:

```
output/
├── shar/                    # Lhotse Shar archives
│   ├── cuts.000000.tar
│   ├── cuts.000001.tar
│   └── ...
├── manifest.jsonl.gz        # CutSet manifest
└── stats.json               # Export statistics
```

## Filtering Criteria

| Filter | Threshold | Description |
|--------|-----------|-------------|
| Duration | 0.5s – 30s | Segment length |
| SNR | >= 10 dB | Signal-to-noise ratio |
| Speech Ratio | >= 50% | VAD-based speech activity |
| CER (manual subs) | <= 15% | Character Error Rate |
| CER (auto subs) | <= 10% | Stricter for auto-generated |

## API Documentation

When the server is running, interactive API docs are available at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## License

MIT License
