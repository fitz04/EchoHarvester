"""Tests for retry utilities."""

import asyncio

import pytest

from echoharvester.utils.retry import async_retry, retry


class TestRetry:
    def test_succeeds_first_try(self):
        call_count = 0

        @retry(max_attempts=3, delay_sec=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_succeeds_after_retries(self):
        call_count = 0

        @retry(max_attempts=3, delay_sec=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        assert fail_twice() == "ok"
        assert call_count == 3

    def test_exhausts_retries(self):
        @retry(max_attempts=2, delay_sec=0.01)
        def always_fail():
            raise ValueError("always")

        with pytest.raises(ValueError, match="always"):
            always_fail()

    def test_specific_exceptions(self):
        @retry(max_attempts=3, delay_sec=0.01, exceptions=(ValueError,))
        def fail_type():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            fail_type()

    def test_on_retry_callback(self):
        callbacks = []

        @retry(
            max_attempts=3,
            delay_sec=0.01,
            on_retry=lambda attempt, e: callbacks.append(attempt),
        )
        def fail_twice():
            if len(callbacks) < 2:
                raise ValueError("retry")
            return "ok"

        fail_twice()
        assert len(callbacks) == 2


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        call_count = 0

        @async_retry(max_attempts=3, delay_sec=0.01)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert await succeed() == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_retries(self):
        call_count = 0

        @async_retry(max_attempts=3, delay_sec=0.01)
        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        assert await fail_twice() == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        @async_retry(max_attempts=2, delay_sec=0.01)
        async def always_fail():
            raise ValueError("always")

        with pytest.raises(ValueError, match="always"):
            await always_fail()
