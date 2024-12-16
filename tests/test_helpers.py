import asyncio
from typing import List

import pytest

from hypellm.helpers import syncify, asyncify, as_completed, gather, amap, pmap
from hypellm.settings import settings


async def async_add(x: int) -> int:
    await asyncio.sleep(0.01)
    return x + 1


def sync_add(x: int) -> int:
    return x + 1


async def async_batch_add(items: List[int]) -> List[int]:
    await asyncio.sleep(0.01)
    return [x + 1 for x in items]


def sync_batch_add(items: List[int]) -> List[int]:
    return [x + 1 for x in items]


def test_syncify():
    sync_fn = syncify(async_add)
    assert sync_fn(1) == 2


@pytest.mark.asyncio
async def test_asyncify():
    async_fn = asyncify(sync_add)
    result = await async_fn(1)
    assert result == 2


@pytest.mark.asyncio
async def test_as_completed():
    tasks = [async_add(i) for i in range(3)]
    results = []

    async for result in as_completed(tasks):
        results.append(result)

    assert sorted(results) == [1, 2, 3]


@pytest.mark.asyncio
async def test_gather():
    tasks = [async_add(i) for i in range(3)]
    results = await gather(tasks)
    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_gather_with_timeout():
    async def slow_task():
        await asyncio.sleep(0.5)
        return 42

    tasks = [slow_task()]
    results = await gather(tasks, timeout=0.1)
    assert results is None


@pytest.mark.asyncio
async def test_amap_single_batch():
    data = [1, 2, 3]
    results = await amap(async_add, data, batch_size=1)
    assert results == [2, 3, 4]


@pytest.mark.asyncio
async def test_amap_multi_batch():
    data = [1, 2, 3, 4]
    results = await amap(async_batch_add, data, batch_size=2)
    assert results == [2, 3, 4, 5]


def test_pmap_single_batch():
    data = [1, 2, 3]
    results = pmap(sync_add, data, batch_size=1)
    assert results == [2, 3, 4]


def test_pmap_multi_batch():
    data = [1, 2, 3, 4]
    results = pmap(sync_batch_add, data, batch_size=2)
    assert results == [2, 3, 4, 5]


@pytest.mark.parametrize(
    "batch_size,concurrency",
    [
        (0, 1),
        (1, 0),
    ],
)
@pytest.mark.asyncio
async def test_amap_invalid_params(batch_size, concurrency):
    with pytest.raises(AssertionError):
        await amap(async_add, [1], batch_size=batch_size, concurrency=concurrency)


@pytest.mark.parametrize(
    "batch_size,concurrency",
    [
        (0, 1),
        (1, 0),
    ],
)
def test_pmap_invalid_params(batch_size, concurrency):
    with pytest.raises(AssertionError):
        pmap(sync_add, [1], batch_size=batch_size, concurrency=concurrency)
