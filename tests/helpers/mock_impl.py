from typing import Optional

from hypellm.helpers import asyncify
from hypellm.types import Datum, Prompt


def inferred_sync(
    data: list[Datum],
    batch_size: int,
    concurrency: Optional[int] = None,
) -> Prompt:
    return Prompt(
        intent="inferred intent",
        dos=["do 1", "do 2"],
        donts=["dont 1", "dont 2"],
        reasoning_steps=["step 1", "step 2"],
        examples=data[:2],
    )


inferred = asyncify(inferred_sync)


def questions_sync(data: Datum) -> list[str]:
    return ["What is the main concept?", "How does this work?", "Why is this important?"]


questions = asyncify(questions_sync)


def reasoned_sync(
    data: list[Datum],
    branching_factor: int = 3,
    concurrency: Optional[int] = None,
) -> list[Datum]:
    return [d.update(reasoning=[f"step {i+1}" for i in range(branching_factor)]) for d in data]


reasoned = asyncify(reasoned_sync)
