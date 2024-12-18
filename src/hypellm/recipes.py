from collections import defaultdict
from typing import Optional

from hypellm.helpers import amap, pmap
from hypellm import settings, Example, Prompt, DataModel


async def inferred(
    data: list[Example],
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Prompt:
    """
    Infer a prompt from a list of datums.

    Args:
        data: List of input/output examples with reasoning steps
        prompt: Optional

    Returns:
        A single Prompt object that could generate the examples
    """
    return await settings.impl.inferred(data, batch_size, concurrency)


def inferred_sync(
    data: list[Example],
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Prompt:
    return settings.impl.inferred_sync(data, batch_size, concurrency)


async def reasoned(
    data: list[Example],
    branching_factor: int = 3,
    concurrency: Optional[int] = None,
    prompt: Optional[Prompt] = None,
) -> tuple[Prompt, list[Example]]:
    """
    Generate hypothetical reasoning steps for a list of input/output examples.

    Args:
        data: List of input/output example pairs to generate reasoning for
        batch_size: Number of examples to process at once
        concurrency: Maximum number of parallel operations
    Returns:
        List of reasoning steps for each example in the input data
    """
    return await settings.impl.reasoned(data, branching_factor, concurrency, prompt)


def reasoned_sync(
    data: list[Example],
    branching_factor: int = 3,
    concurrency: Optional[int] = None,
    prompt: Optional[Prompt] = None,
) -> tuple[Prompt, list[Example]]:
    return settings.impl.reasoned_sync(data, branching_factor, concurrency, prompt)


async def inverted(
    data: list[Example],
    branching_factor: int = 3,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> tuple[Prompt, list[Example]]:
    """
    Given a list of input/output examples, infers a prompt that could have generated those inputs from those outputs.

    This is useful for:
    - Reverse engineering the prompt that would generate demonstrated examples

    Args:
        data: List of input/output example pairs
        batch_size: Number of examples to process at once
        concurrency: Maximum number of parallel operations

    Returns:
        A list of inverted Datum objects with the reasoning steps added
    """
    inverted_data = [Example(inputs=datum.outputs, outputs=datum.inputs) for datum in data]
    prompt = await inferred(inverted_data, batch_size, concurrency)
    return await reasoned(inverted_data, branching_factor, concurrency, prompt)


def inverted_sync(
    data: list[Example],
    branching_factor: int = 3,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> tuple[Prompt, list[Example]]:
    inverted_data = [Example(inputs=datum.outputs, outputs=datum.inputs) for datum in data]
    prompt = inferred_sync(inverted_data, batch_size, concurrency)
    return reasoned_sync(inverted_data, branching_factor, concurrency, prompt)


async def questions(
    data: list[DataModel],
    concurrency: Optional[int] = None,
) -> dict[str, list[DataModel]]:
    """
    Given a list of data, infers questions that can be answered by the data.

    This is useful for:
    - Improving lookup performance for a dataset

    Args:
        data: List of data
        batch_size: Number of examples to process at once
        concurrency: Maximum number of parallel operations

    Returns:
        A dictionary mapping each question to a list of data that contain the answer
    """
    response = defaultdict(list)

    for qs, answer in zip(
        await amap(settings.impl.questions, data, batch_size=1, concurrency=concurrency),
        data,
    ):
        for q in qs:
            response[q].append(answer)

    return response


def questions_sync(
    data: list[DataModel],
    concurrency: Optional[int] = None,
) -> dict[str, list[DataModel]]:
    response = defaultdict(list)

    for qs, answer in zip(
        pmap(settings.impl.questions_sync, data, batch_size=1, concurrency=concurrency),
        data,
    ):
        for q in qs:
            response[q].append(answer)

    return response
