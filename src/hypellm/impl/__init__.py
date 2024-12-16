from typing import Optional, Protocol

from hypellm.types import Datum, Prompt


class Impl(Protocol):
    @staticmethod
    async def inferred(
        data: list[Datum], batch_size: Optional[int] = None, concurrency: Optional[int] = None
    ) -> Prompt: ...

    @staticmethod
    def inferred_sync(
        data: list[Datum], batch_size: Optional[int] = None, concurrency: Optional[int] = None
    ) -> Prompt: ...

    @staticmethod
    async def questions(data: list[Datum]) -> list[str]: ...

    @staticmethod
    def questions_sync(data: list[Datum]) -> list[str]: ...

    @staticmethod
    async def reasoned(
        data: list[Datum],
        branching_factor: int = 3,
        concurrency: Optional[int] = None,
        prompt: Optional[Prompt] = None,
    ) -> tuple[Prompt, list[Datum]]: ...

    @staticmethod
    def reasoned_sync(
        data: list[Datum],
        branching_factor: int = 3,
        concurrency: Optional[int] = None,
        prompt: Optional[Prompt] = None,
    ) -> tuple[Prompt, list[Datum]]: ...
