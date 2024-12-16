import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Callable, Optional, Coroutine, Any, Union

import anyio
import asyncer

try:
    import tqdm
    import tqdm.asyncio
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
    abandon_on_cancel: bool = False,
    cancellable: Union[bool, None] = None,
    limiter: Optional[anyio.CapacityLimiter] = None,
) -> Callable[T_ParamSpec, Coroutine[Any, Any, T_Retval]]:
    return asyncer.asyncify(sync_function, abandon_on_cancel, cancellable, limiter)


async def as_completed(
    tasks: list[Coroutine[Any, Any, T_Retval]],
    *,
    timeout: Optional[float] = None,
) -> AsyncIterator[tuple[int, T_Retval]]:
    if settings.show_progress and tqdm is not None:
        return tqdm.asyncio.as_completed(tasks, timeout=timeout, total=len(tasks))
    else:
        return asyncio.as_completed(tasks, timeout=timeout)


async def gather(
    tasks: list[Coroutine[Any, Any, T_Retval]],
    *,
    timeout: Optional[float] = None,
) -> list[T_Retval]:
    if settings.show_progress and tqdm is not None:
        return tqdm.asyncio.gather(tasks, timeout=timeout, total=len(tasks))
    else:
        return asyncio.gather(tasks, timeout=timeout)


async def amap(
    func: Callable[T_ParamSpec, Coroutine[Any, Any, T_Retval]],
    data: list[Datum],
    batch_size: int = 5,
    concurrency: int = 10,
) -> list[T_Retval]:
    assert batch_size > 0, "batch_size must be greater than 0"
    assert concurrency > 0, "concurrency must be greater than 0"

    semaphore = anyio.Semaphore(concurrency)

    if batch_size == 1:

        async def handle_batch(batch: list[Datum]) -> list[T_Retval]:
            async with semaphore:
                result = await func(batch[0])
                return [result]
    else:

        async def handle_batch(batch: list[Datum]) -> list[T_Retval]:
            async with semaphore:
                results = await func(batch)
                return results

    results = await gather(
        [handle_batch(data[i : i + batch_size]) for i in range(0, len(data), batch_size)]
    )
    return [result for batch in results for result in batch]


def pmap(
    func: Callable[T_ParamSpec, T_Retval],
    data: list[Datum],
    batch_size: int = 5,
    concurrency: int = 10,
) -> list[T_Retval]:
    assert batch_size > 0, "batch_size must be greater than 0"
    assert concurrency > 0, "concurrency must be greater than 0"

    if batch_size == 1:

        def handle_batch(batch: list[Datum]) -> list[T_Retval]:
            return [func(batch[0])]

    else:

        def handle_batch(batch: list[Datum]) -> list[T_Retval]:
            return func(batch)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        batch_tasks = [
            executor.submit(handle_batch, data[i : i + batch_size])
            for i in range(0, len(data), batch_size)
        ]

        if settings.show_progress and tqdm is not None:
            futures = tqdm.tqdm(batch_tasks, total=len(batch_tasks))
        else:
            futures = batch_tasks

        results = [f.result() for f in futures]

    return [result for batch in results for result in batch]
