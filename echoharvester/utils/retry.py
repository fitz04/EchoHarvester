"""Retry utilities for transient failure handling."""

import asyncio
import functools
import logging
import time
from typing import Callable, Type

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay_sec: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable | None = None,
):
    """Decorator for retrying a synchronous function on transient failures.

    Args:
        max_attempts: Maximum number of attempts
        delay_sec: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(attempt, exception) called before each retry
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = delay_sec

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    if on_retry:
                        on_retry(attempt, e)
                    time.sleep(delay)
                    delay *= backoff_factor

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay_sec: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable | None = None,
):
    """Decorator for retrying an async function on transient failures.

    Args:
        max_attempts: Maximum number of attempts
        delay_sec: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(attempt, exception) called before each retry
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = delay_sec

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    if on_retry:
                        on_retry(attempt, e)
                    await asyncio.sleep(delay)
                    delay *= backoff_factor

        return wrapper

    return decorator
