"""Base stage class for pipeline stages."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from echoharvester.config import Config
from echoharvester.db import Database

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """Abstract base class for pipeline stages."""

    name: str = "base"

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self._running = False
        self._cancelled = False

    @abstractmethod
    async def process(self) -> AsyncGenerator[dict[str, Any], None]:
        """Process items and yield results.

        Yields:
            Dictionary with processing results
        """
        pass

    async def run(self, run_id: int) -> dict[str, Any]:
        """Run the stage and track progress.

        Args:
            run_id: Pipeline run ID for progress tracking

        Returns:
            Stage statistics
        """
        self._running = True
        self._cancelled = False

        processed = 0
        passed = 0
        failed = 0
        errors = []
        start_time = time.time()

        logger.info(f"[{self.name}] Stage started (run_id={run_id})")

        try:
            async for result in self.process():
                if self._cancelled:
                    logger.info(f"Stage {self.name} cancelled")
                    break

                processed += 1

                if result.get("status") == "passed":
                    passed += 1
                elif result.get("status") == "failed":
                    failed += 1
                elif result.get("status") == "error":
                    errors.append(result.get("error"))

                # Update progress
                await self.db.update_progress(
                    run_id=run_id,
                    stage=self.name,
                    processed=processed,
                    total=result.get("total", processed),
                    current_item=result.get("item_id", ""),
                )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{self.name}] Stage failed after {elapsed:.1f}s: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise
        finally:
            self._running = False

        elapsed = time.time() - start_time
        logger.info(
            f"[{self.name}] Stage completed in {elapsed:.1f}s: "
            f"processed={processed}, passed={passed}, "
            f"failed={failed}, errors={len(errors)}"
        )

        return {
            "stage": self.name,
            "processed": processed,
            "passed": passed,
            "failed": failed,
            "errors": len(errors),
            "elapsed_sec": round(elapsed, 1),
        }

    def cancel(self):
        """Cancel the running stage."""
        self._cancelled = True

    @property
    def is_running(self) -> bool:
        return self._running
