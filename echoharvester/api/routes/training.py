"""Training API routes."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass
class TrainingState:
    """Mutable training state stored on app.state."""

    state: str = "idle"  # idle, preparing, training, decoding, exporting
    trainer: object | None = None
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    lr: float = 0.0
    epochs: list[dict] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    error: str | None = None
    # For decode progress
    decode_processed: int = 0
    decode_total: int = 0


def _get_training_config(request: Request):
    """Get TrainingConfig from app config."""
    config = request.app.state.config
    tc = config.get_training_config()
    if tc is None:
        raise HTTPException(status_code=400, detail="Training not configured in config.yaml")
    return tc


def _get_state(request: Request) -> TrainingState:
    """Get or initialize TrainingState on app."""
    if not hasattr(request.app.state, "training_state"):
        request.app.state.training_state = TrainingState()
    return request.app.state.training_state


def _get_training_manager(request: Request):
    """Get training WebSocket manager."""
    from echoharvester.api.routes.websocket import training_manager
    return training_manager


def _make_progress_callback(request: Request):
    """Create a progress callback that bridges thread -> async WebSocket broadcast."""
    manager = _get_training_manager(request)
    ts = _get_state(request)
    loop = asyncio.get_event_loop()

    def callback(event: dict):
        # Update TrainingState based on event
        evt = event.get("event")
        if evt == "train_start":
            ts.total_epochs = event.get("total_epochs", 0)
        elif evt == "epoch_start":
            ts.current_epoch = event.get("epoch", 0)
        elif evt == "batch_update":
            ts.train_loss = event.get("loss", 0)
            ts.lr = event.get("lr", 0)
        elif evt == "epoch_complete":
            ts.train_loss = event.get("train_loss", 0)
            ts.val_loss = event.get("val_loss") or 0
            ts.lr = event.get("lr", 0)
            ts.epochs.append(event)
            if event.get("is_best"):
                ts.best_val_loss = ts.val_loss
                ts.best_epoch = event.get("epoch", 0)
        elif evt == "decode_progress":
            ts.decode_processed = event.get("processed", 0)

        # Broadcast to WebSocket clients from thread
        try:
            asyncio.run_coroutine_threadsafe(manager.broadcast(event), loop)
        except Exception:
            pass

    return callback


@router.get("/config")
async def get_config(request: Request):
    """Get current training config + filesystem state."""
    try:
        tc = _get_training_config(request)
    except HTTPException:
        return {"configured": False}

    return {
        "configured": True,
        "shar_sources": tc.shar_sources,
        "data_dir": str(tc.data_dir),
        "exp_dir": str(tc.exp_dir),
        "model": tc.model.model_dump(mode="json"),
        "training_params": tc.training_params.model_dump(mode="json"),
        "device": tc.device,
    }


@router.get("/data-stats")
async def get_data_stats(request: Request):
    """Return training_data/stats.json if it exists."""
    tc = _get_training_config(request)
    stats_path = Path(tc.data_dir) / "stats.json"
    if not stats_path.exists():
        return {"prepared": False}

    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    return {"prepared": True, **stats}


@router.post("/prepare")
async def prepare_data(request: Request):
    """Run DataPreparer + tokenizer in background thread."""
    tc = _get_training_config(request)
    ts = _get_state(request)

    if ts.state != "idle":
        raise HTTPException(status_code=409, detail=f"Cannot prepare: state is '{ts.state}'")

    ts.state = "preparing"
    ts.error = None
    callback = _make_progress_callback(request)

    async def run():
        try:
            from echoharvester.training.data_prep import DataPreparer

            preparer = DataPreparer(tc, progress_callback=callback)
            stats = await asyncio.to_thread(preparer.prepare)

            # Build tokenizer
            callback({"event": "prepare_progress", "step": "tokenizer", "detail": "Building tokenizer..."})
            from echoharvester.training.tokenizer import build_tokenizer
            tokenizer = await asyncio.to_thread(build_tokenizer, tc)
            callback({"event": "prepare_progress", "step": "tokenizer_done",
                       "detail": f"Tokenizer built: {tokenizer.vocab_size} tokens"})

            callback({"event": "prepare_complete", "stats": stats})
        except Exception as e:
            logger.exception("Data preparation failed")
            ts.error = str(e)
            callback({"event": "prepare_error", "error": str(e)})
        finally:
            ts.state = "idle"

    asyncio.create_task(run())
    return {"status": "started"}


@router.get("/checkpoints")
async def list_checkpoints(request: Request):
    """List checkpoints in exp/ directory."""
    tc = _get_training_config(request)
    exp_dir = Path(tc.exp_dir)

    checkpoints = []
    if exp_dir.exists():
        for pt_file in sorted(exp_dir.glob("*.pt")):
            size_mb = pt_file.stat().st_size / (1024 * 1024)
            checkpoints.append({
                "name": pt_file.name,
                "path": str(pt_file),
                "size_mb": round(size_mb, 1),
                "is_best": pt_file.name == "best.pt",
            })

    return {"checkpoints": checkpoints}


@router.get("/status")
async def get_status(request: Request):
    """Get current training state."""
    ts = _get_state(request)
    return {
        "state": ts.state,
        "current_epoch": ts.current_epoch,
        "total_epochs": ts.total_epochs,
        "train_loss": ts.train_loss,
        "val_loss": ts.val_loss,
        "lr": ts.lr,
        "best_val_loss": ts.best_val_loss if ts.best_val_loss != float("inf") else None,
        "best_epoch": ts.best_epoch,
        "error": ts.error,
        "epochs": ts.epochs,
    }


@router.post("/start")
async def start_training(request: Request):
    """Start training in background thread."""
    tc = _get_training_config(request)
    ts = _get_state(request)

    if ts.state != "idle":
        raise HTTPException(status_code=409, detail=f"Cannot start: state is '{ts.state}'")

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    resume_checkpoint = body.get("resume_checkpoint")
    pretrained_checkpoint = body.get("pretrained_checkpoint")
    freeze_encoder_epochs = int(body.get("freeze_encoder_epochs", 0))
    epochs_override = body.get("num_epochs")

    if epochs_override:
        tc.training_params.num_epochs = int(epochs_override)

    # Apply training param overrides from UI
    if "lr_factor" in body:
        tc.training_params.lr_factor = float(body["lr_factor"])
    if "warmup_epochs" in body:
        tc.training_params.warmup_epochs = float(body["warmup_epochs"])
    if "warm_step" in body:
        tc.training_params.warm_step = int(body["warm_step"])

    ts.state = "training"
    ts.error = None
    ts.epochs = []
    ts.current_epoch = 0
    ts.best_val_loss = float("inf")
    ts.best_epoch = 0
    callback = _make_progress_callback(request)

    async def run():
        try:
            from echoharvester.training.trainer import Trainer

            trainer = Trainer(tc, progress_callback=callback)
            ts.trainer = trainer
            stats = await asyncio.to_thread(
                trainer.train,
                resume_checkpoint=resume_checkpoint,
                pretrained_checkpoint=pretrained_checkpoint,
                freeze_encoder_epochs=freeze_encoder_epochs,
            )
            ts.best_val_loss = stats.get("best_val_loss", float("inf"))
            ts.best_epoch = stats.get("best_epoch", 0)
        except Exception as e:
            logger.exception("Training failed")
            ts.error = str(e)
            callback({"event": "train_error", "error": str(e)})
        finally:
            ts.state = "idle"
            ts.trainer = None

    asyncio.create_task(run())
    return {"status": "started"}


@router.post("/stop")
async def stop_training(request: Request):
    """Stop training at next epoch boundary."""
    ts = _get_state(request)

    if ts.state != "training":
        raise HTTPException(status_code=409, detail="Not currently training")

    if ts.trainer and hasattr(ts.trainer, "stop"):
        ts.trainer.stop()
        return {"status": "stop_requested"}

    raise HTTPException(status_code=500, detail="Trainer instance not available")


@router.post("/decode")
async def start_decode(request: Request):
    """Run decoding in background thread."""
    tc = _get_training_config(request)
    ts = _get_state(request)

    if ts.state != "idle":
        raise HTTPException(status_code=409, detail=f"Cannot decode: state is '{ts.state}'")

    body = await request.json()
    checkpoint = body.get("checkpoint")
    split = body.get("split", "test")

    if not checkpoint:
        raise HTTPException(status_code=400, detail="checkpoint is required")

    ts.state = "decoding"
    ts.error = None
    ts.decode_processed = 0
    callback = _make_progress_callback(request)

    async def run():
        try:
            from echoharvester.training.decode import Decoder

            decoder = Decoder(tc, progress_callback=callback)
            await asyncio.to_thread(decoder.decode, checkpoint, split, 0)
        except Exception as e:
            logger.exception("Decoding failed")
            ts.error = str(e)
            callback({"event": "decode_error", "error": str(e)})
        finally:
            ts.state = "idle"

    asyncio.create_task(run())
    return {"status": "started"}


@router.get("/decode-results")
async def get_decode_results(request: Request):
    """Return decode results JSON."""
    tc = _get_training_config(request)
    split = request.query_params.get("split", "test")
    results_path = Path(tc.exp_dir) / f"decode_{split}.json"

    if not results_path.exists():
        return {"available": False}

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    return {"available": True, **results}


@router.get("/decode-results/audio/{idx}")
async def get_decode_audio(request: Request, idx: int):
    """Serve audio for a decoded sample by index."""
    tc = _get_training_config(request)
    split = request.query_params.get("split", "test")

    # Load the corresponding cut from the split
    cuts_path = Path(tc.data_dir) / f"{split}_cuts.jsonl.gz"
    if not cuts_path.exists():
        raise HTTPException(status_code=404, detail=f"Cuts file not found: {cuts_path}")

    try:
        from lhotse import CutSet
        cs = CutSet.from_jsonl(cuts_path)
        cuts_list = list(cs)

        if idx < 0 or idx >= len(cuts_list):
            raise HTTPException(status_code=404, detail=f"Index {idx} out of range (0-{len(cuts_list)-1})")

        cut = cuts_list[idx]
        audio_path = cut.recording.sources[0].source
        if not Path(audio_path).exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

        return FileResponse(audio_path, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_model(request: Request):
    """Export model in background thread."""
    tc = _get_training_config(request)
    ts = _get_state(request)

    if ts.state != "idle":
        raise HTTPException(status_code=409, detail=f"Cannot export: state is '{ts.state}'")

    body = await request.json()
    checkpoint = body.get("checkpoint")
    fmt = body.get("format", "onnx")

    if not checkpoint:
        raise HTTPException(status_code=400, detail="checkpoint is required")

    ts.state = "exporting"
    ts.error = None
    callback = _make_progress_callback(request)

    async def run():
        try:
            from echoharvester.training.export import ModelExporter

            exporter = ModelExporter(tc, progress_callback=callback)
            await asyncio.to_thread(exporter.export, checkpoint, None, fmt)
        except Exception as e:
            logger.exception("Export failed")
            ts.error = str(e)
            callback({"event": "export_error", "error": str(e)})
        finally:
            ts.state = "idle"

    asyncio.create_task(run())
    return {"status": "started"}
