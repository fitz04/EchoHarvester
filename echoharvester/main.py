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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
