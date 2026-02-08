"""Main entry point for EchoHarvester CLI."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm


def setup_logging(level: str = "INFO"):
    """Setup logging configuration.

    Only configures the root logger at WARNING to suppress third-party noise.
    The echoharvester logger is configured separately in Config._setup_logging().
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # echoharvester logger level is controlled by config; set a CLI default
    # that will be overridden by Config._setup_logging() when config loads.
    eh_logger = logging.getLogger("echoharvester")
    eh_logger.setLevel(getattr(logging, level.upper()))


async def run_pipeline(args):
    """Run the pipeline."""
    from echoharvester.config import load_config
    from echoharvester.pipeline import PipelineOrchestrator

    config = load_config(args.config)
    orchestrator = PipelineOrchestrator(config)

    # Setup progress bar
    pbar = None

    def progress_callback(data: dict):
        nonlocal pbar
        event = data.get("event")

        if event == "stage_start":
            print(f"\n>>> Stage: {data.get('stage')}")

        elif event == "stage_complete":
            stats = data.get("stats", {})
            print(f"    Processed: {stats.get('processed', 0)}")
            print(f"    Passed: {stats.get('passed', 0)}")

        elif event == "pipeline_complete":
            print(f"\n=== Pipeline Complete ===")
            print(f"State: {data.get('state')}")

    orchestrator.set_progress_callback(progress_callback)

    # Determine stages to run
    stages = None
    if args.stage:
        stages = [args.stage]

    try:
        stats = await orchestrator.run(stages=stages)
        print(f"\nFinal stats: {json.dumps(stats, indent=2, default=str)}")
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping...")
        orchestrator.stop()
    except Exception as e:
        print(f"\nError: {e}")
        raise


async def show_status(args):
    """Show pipeline status."""
    from echoharvester.config import load_config
    from echoharvester.db import get_db

    config = load_config(args.config)
    db = await get_db()

    stats = await db.get_stats()
    print("\n=== Pipeline Statistics ===")
    print(f"Total sources: {stats.get('total_sources', 0)}")
    print(f"Total media items: {stats.get('total_media_items', 0)}")
    print(f"Total segments: {stats.get('total_segments', 0)}")
    print(f"Passed duration: {stats.get('passed_duration_hours', 0):.2f} hours")
    print(f"Average CER: {stats.get('avg_cer', 0):.4f}")

    print("\nSegment status breakdown:")
    for status, count in stats.get("segment_status", {}).items():
        print(f"  {status}: {count}")

    print("\nCER distribution:")
    for bucket, count in stats.get("cer_distribution", {}).items():
        print(f"  {bucket}: {count}")


async def show_errors(args):
    """Show error list."""
    from echoharvester.config import load_config
    from echoharvester.db import Status, get_db

    config = load_config(args.config)
    db = await get_db()

    # Get errored media items
    items = await db.get_media_items(status=Status.ERROR.value, limit=50)

    print("\n=== Error List ===")
    for item in items:
        print(f"\n[{item['id']}] {item.get('title', 'Unknown')}")
        print(f"  Error: {item.get('error_msg', 'No message')}")


async def generate_report(args):
    """Generate statistics report."""
    from echoharvester.config import load_config
    from echoharvester.db import get_db

    config = load_config(args.config)
    db = await get_db()

    stats = await db.get_stats()
    recent = await db.get_recent_segments(limit=10)

    report = {
        "summary": stats,
        "recent_segments": recent,
    }

    output_path = config.paths.output_dir / "report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"Report saved to: {output_path}")


def run_server(args):
    """Run the web server."""
    from echoharvester.config import load_config

    config = load_config(args.config)

    import uvicorn

    from echoharvester.api.app import create_app

    app = create_app(config)

    print(f"\nStarting web server at http://{config.web.host}:{config.web.port}")

    uvicorn.run(
        app,
        host=config.web.host,
        port=config.web.port,
        reload=config.web.reload,
    )


def train_prepare(args):
    """Prepare training data: load shar, split, build tokenizer."""
    from echoharvester.config import load_config

    config = load_config(args.config)
    tc = config.get_training_config()
    if tc is None:
        print("Error: 'training' section not found in config.yaml")
        sys.exit(1)

    from echoharvester.training.data_prep import DataPreparer
    from echoharvester.training.tokenizer import build_tokenizer

    # Step 1: Data preparation (load, merge, split)
    preparer = DataPreparer(tc)
    stats = preparer.prepare()
    print(f"\nData preparation complete:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Step 2: Build tokenizer
    tokenizer = build_tokenizer(tc)
    print(f"\nTokenizer built: {tokenizer.vocab_size} tokens")


def train_run(args):
    """Run model training."""
    from echoharvester.config import load_config

    config = load_config(args.config)
    tc = config.get_training_config()
    if tc is None:
        print("Error: 'training' section not found in config.yaml")
        sys.exit(1)

    # Override epochs if specified
    if args.epochs:
        tc.training_params.num_epochs = args.epochs

    from echoharvester.training.trainer import Trainer

    trainer = Trainer(tc)
    stats = trainer.train(
        resume_checkpoint=args.resume,
        pretrained_checkpoint=args.pretrained,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
    )
    print(f"\nTraining complete:")
    print(f"  Best val_loss: {stats['best_val_loss']} (epoch {stats['best_epoch']})")


def train_status(args):
    """Show training status."""
    from echoharvester.config import load_config

    config = load_config(args.config)
    tc = config.get_training_config()
    if tc is None:
        print("Error: 'training' section not found in config.yaml")
        sys.exit(1)

    exp_dir = Path(tc.exp_dir)

    # Check for training stats
    stats_path = exp_dir / "training_stats.json"
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        print("\n=== Training Status ===")
        print(f"Best val_loss: {stats.get('best_val_loss', 'N/A')}")
        print(f"Best epoch: {stats.get('best_epoch', 'N/A')}")
        epochs = stats.get("epochs", [])
        if epochs:
            last = epochs[-1]
            print(f"Last epoch: {last['epoch']}")
            print(f"  train_loss: {last['train_loss']}")
            print(f"  val_loss: {last.get('val_loss', 'N/A')}")
    else:
        print("No training stats found. Run 'train run' first.")

    # List checkpoints
    checkpoints = sorted(exp_dir.glob("epoch-*.pt"))
    if checkpoints:
        print(f"\nCheckpoints ({len(checkpoints)}):")
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  {ckpt.name} ({size_mb:.1f} MB)")
        if (exp_dir / "best.pt").exists():
            print("  best.pt (symlink)")
    else:
        print("\nNo checkpoints found.")

    # Check data preparation
    data_dir = Path(tc.data_dir)
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}_cuts.jsonl.gz"
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"\n  {split}: {path} ({size_mb:.1f} MB)")

    tokens_path = data_dir / "lang_char" / "tokens.txt"
    if tokens_path.exists():
        num_tokens = sum(1 for _ in open(tokens_path, encoding="utf-8"))
        print(f"  Tokenizer: {num_tokens} tokens")


def train_decode(args):
    """Decode test set and compute CER."""
    from echoharvester.config import load_config

    config = load_config(args.config)
    tc = config.get_training_config()
    if tc is None:
        print("Error: 'training' section not found in config.yaml")
        sys.exit(1)

    from echoharvester.training.decode import Decoder

    decoder = Decoder(tc)
    results = decoder.decode(
        checkpoint_path=args.checkpoint,
        split=args.split,
    )
    print(f"\nDecode results:")
    print(f"  Split: {results['split']}")
    print(f"  Utterances: {results['num_utterances']}")
    print(f"  CER: {results['overall_cer']:.4f}")


def train_export(args):
    """Export model to ONNX or TorchScript."""
    from echoharvester.config import load_config

    config = load_config(args.config)
    tc = config.get_training_config()
    if tc is None:
        print("Error: 'training' section not found in config.yaml")
        sys.exit(1)

    from echoharvester.training.export import ModelExporter

    exporter = ModelExporter(tc)
    output = exporter.export(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        format=args.format,
    )
    print(f"\nModel exported to: {output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EchoHarvester - Korean ASR Training Data Pipeline"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Configuration file path",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument(
        "--stage",
        "-s",
        choices=["metadata", "download", "preprocess", "validate", "export"],
        help="Run specific stage only",
    )

    # Status command
    subparsers.add_parser("status", help="Show pipeline status")

    # Errors command
    subparsers.add_parser("errors", help="Show error list")

    # Report command
    subparsers.add_parser("report", help="Generate statistics report")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run web server")
    server_parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to",
    )

    # Train command group
    train_parser = subparsers.add_parser("train", help="Training commands")
    train_subparsers = train_parser.add_subparsers(
        dest="train_command", help="Training sub-commands"
    )

    # train prepare
    train_subparsers.add_parser("prepare", help="Prepare training data (split + tokenizer)")

    # train run
    train_run_parser = train_subparsers.add_parser("run", help="Run model training")
    train_run_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    train_run_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    train_run_parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained checkpoint for fine-tuning",
    )
    train_run_parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="Number of epochs to freeze encoder (0 = no freeze)",
    )

    # train status
    train_subparsers.add_parser("status", help="Show training status")

    # train decode
    train_decode_parser = train_subparsers.add_parser(
        "decode", help="Decode test set and compute CER"
    )
    train_decode_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    train_decode_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Data split to decode",
    )

    # train export
    train_export_parser = train_subparsers.add_parser(
        "export", help="Export model to ONNX/TorchScript"
    )
    train_export_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    train_export_parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript"],
        help="Export format",
    )
    train_export_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    # Run command
    if args.command == "run":
        asyncio.run(run_pipeline(args))
    elif args.command == "status":
        asyncio.run(show_status(args))
    elif args.command == "errors":
        asyncio.run(show_errors(args))
    elif args.command == "report":
        asyncio.run(generate_report(args))
    elif args.command == "server":
        run_server(args)
    elif args.command == "train":
        if args.train_command == "prepare":
            train_prepare(args)
        elif args.train_command == "run":
            train_run(args)
        elif args.train_command == "status":
            train_status(args)
        elif args.train_command == "decode":
            train_decode(args)
        elif args.train_command == "export":
            train_export(args)
        else:
            train_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
