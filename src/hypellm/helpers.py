import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Callable, Optional, Coroutine, Any, Union

import anyio
import asyncer

try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    tqdm = None

from hypellm.settings import settings
from hypellm.types import Datum, T_ParamSpec, T_Retval


def syncify(
    async_function: Callable[T_ParamSpec, Coroutine[Any, Any, T_Retval]],
    raise_sync_error: bool = False,
) -> Callable[T_ParamSpec, T_Retval]:
    return asyncer.syncify(async_function, raise_sync_error)


def asyncify(
    sync_function: Callable[T_ParamSpec, T_Retval],
    *,
    cancellable: Union[bool, None] = None,
    limiter: Optional[anyio.CapacityLimiter] = None,
) -> Callable[T_ParamSpec, Coroutine[Any, Any, T_Retval]]:
    return asyncer.asyncify(sync_function, cancellable=cancellable, limiter=limiter)


async def as_completed(
    tasks: list[Coroutine[Any, Any, T_Retval]],
    *,
    timeout: Optional[float] = None,
) -> AsyncIterator[tuple[int, T_Retval]]:
    if settings.show_progress and tqdm_asyncio is not None:
        generator = tqdm_asyncio.as_completed(tasks, timeout=timeout, total=len(tasks))
    else:
        generator = asyncio.as_completed(tasks, timeout=timeout)

    for coro in generator:
        yield await coro


async def gather(
    tasks: list[Coroutine[Any, Any, T_Retval]],
    *,
    timeout: Optional[float] = None,
) -> list[T_Retval]:
    with anyio.move_on_after(timeout):
        if settings.show_progress and tqdm_asyncio is not None:
            return await tqdm_asyncio.gather(*tasks, total=len(tasks))
        else:
            return await asyncio.gather(*tasks)


async def amap(
    func: Callable[T_ParamSpec, Coroutine[Any, Any, T_Retval]],
    data: list[Datum],
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> list[T_Retval]:
    if batch_size is None:
        batch_size = settings.batch_size
    if concurrency is None:
        concurrency = settings.concurrency

    assert batch_size > 0, "batch_size must be greater than 0"
    assert concurrency > 0, "concurrency must be greater than 0"

    semaphore = anyio.Semaphore(concurrency)

    if batch_size == 1:

        async def handle_batch(batch: list[Datum]) -> T_Retval:
            async with semaphore:
                return await func(batch[0])
    else:

        async def handle_batch(batch: list[Datum]) -> T_Retval:
            async with semaphore:
                return await func(batch)

    return await gather(
        [handle_batch(data[i : i + batch_size]) for i in range(0, len(data), batch_size)]
    )


def pmap(
    func: Callable[T_ParamSpec, T_Retval],
    data: list[Datum],
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> list[T_Retval]:
    if batch_size is None:
        batch_size = settings.batch_size
    if concurrency is None:
        concurrency = settings.concurrency

    assert batch_size > 0, "batch_size must be greater than 0"
    assert concurrency > 0, "concurrency must be greater than 0"

    if batch_size == 1:

        def handle_batch(batch: list[Datum]) -> T_Retval:
            return func(batch[0])

    else:

        def handle_batch(batch: list[Datum]) -> T_Retval:
            return func(batch)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        batch_tasks = [
            pool.submit(handle_batch, data[i : i + batch_size])
            for i in range(0, len(data), batch_size)
        ]

        if settings.show_progress and tqdm is not None:
            futures = tqdm(batch_tasks, total=len(batch_tasks))
        else:
            futures = batch_tasks

        return [f.result() for f in futures]
