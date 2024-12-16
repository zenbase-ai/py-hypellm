from typing import Optional
from hypellm.types import Datum


async def inferred(
    data: list[Datum],
    batch_size: int,
    concurrency: Optional[int] = None,
) -> list[Datum]:
    return data


def inferred_sync(
    data: list[Datum],
    batch_size: int,
    concurrency: Optional[int] = None,
) -> list[Datum]:
    return ...


async def questions(data: list[Datum]) -> list[str]:
    return data


def questions_sync(data: list[Datum]) -> list[str]:
    return ...


async def reasoned(
    data: list[Datum],
    branching_factor: int = 3,
    concurrency: Optional[int] = None,
) -> list[Datum]:
    return data


def reasoned_sync(
    data: list[Datum],
    branching_factor: int = 3,
    concurrency: Optional[int] = None,
) -> list[Datum]:
    return ...
