from typing import Optional, Protocol

from hypellm.types import Example, Prompt


class Impl(Protocol):
    @staticmethod
    async def inferred(
        data: list[Example], batch_size: Optional[int] = None, concurrency: Optional[int] = None
    ) -> Prompt: ...

    @staticmethod
    def inferred_sync(
        data: list[Example], batch_size: Optional[int] = None, concurrency: Optional[int] = None
    ) -> Prompt: ...

    @staticmethod
    async def questions(data: list[Example]) -> list[str]: ...

    @staticmethod
    def questions_sync(data: list[Example]) -> list[str]: ...

    @staticmethod
    async def reasoned(
        data: list[Example],
        branching_factor: int = 3,
        concurrency: Optional[int] = None,
        prompt: Optional[Prompt] = None,
    ) -> tuple[Prompt, list[Example]]: ...

    @staticmethod
    def reasoned_sync(
        data: list[Example],
        branching_factor: int = 3,
        concurrency: Optional[int] = None,
        prompt: Optional[Prompt] = None,
    ) -> tuple[Prompt, list[Example]]: ...
