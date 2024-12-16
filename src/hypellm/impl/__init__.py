from typing import Optional, Protocol

from hypellm.types import Result, Prompt


class Impl(Protocol):
    @staticmethod
    async def inferred(
        data: list[Result], batch_size: Optional[int] = None, concurrency: Optional[int] = None
    ) -> Prompt: ...

    @staticmethod
    def inferred_sync(
        data: list[Result], batch_size: Optional[int] = None, concurrency: Optional[int] = None
    ) -> Prompt: ...

    @staticmethod
    async def questions(data: list[Result]) -> list[str]: ...

    @staticmethod
    def questions_sync(data: list[Result]) -> list[str]: ...

    @staticmethod
    async def reasoned(
        data: list[Result],
        branching_factor: int = 3,
        concurrency: Optional[int] = None,
        prompt: Optional[Prompt] = None,
    ) -> tuple[Prompt, list[Result]]: ...

    @staticmethod
    def reasoned_sync(
        data: list[Result],
        branching_factor: int = 3,
        concurrency: Optional[int] = None,
        prompt: Optional[Prompt] = None,
    ) -> tuple[Prompt, list[Result]]: ...
