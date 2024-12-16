from functools import partial
from random import sample
from typing import Optional

from hypellm import settings
from hypellm.helpers import amap
from hypellm.types import Datum, Prompt

from .base import client, reasoned_model


async def inferred(
    data: list[Datum],
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Prompt:
    """
    Infer a prompt from a list of datums.

    This function:
    1. Splits the data into batches
    2. Infers candidate prompts for each batch in parallel
    3. Recursively combines the candidate prompts until a single prompt remains

    Args:
        data: List of input/output examples with reasoning steps
        prompt: Optional

    Returns:
        A single Prompt object that could generate the examples
    """
    batch_size = batch_size or settings.batch_size
    concurrency = concurrency or settings.concurrency

    candidates = await amap(infer_prompt, data, batch_size, concurrency)
    while len(candidates) > 1:
        candidates = await amap(
            partial(combine_prompts, data),
            candidates,
            batch_size,
            concurrency,
        )

    return candidates[0]


SYSTEM_PROMPT = Prompt(
    intent="Generate a prompt that could generate examples like the ones provided.",
    dos=[
        "Analyze the input/output patterns carefully",
        "Identify key transformations and rules",
        "Create clear, specific instructions",
        "Include both general principles and specific requirements",
        "Ensure the prompt covers all edge cases in examples",
    ],
    donts=[
        "Don't make assumptions beyond what's shown in examples",
        "Don't include contradictory instructions",
        "Don't be overly specific to single examples",
        "Don't ignore important patterns in the data",
    ],
    reasoning_steps=[
        "Examine all input/output pairs to understand the task",
        "Identify common patterns and transformations",
        "Note any special cases or exceptions",
        "Formulate clear instructions that would produce these outputs",
    ],
)


async def infer_prompt(examples: list[Datum]) -> Prompt:
    return await client.chat.completions.create(
        response_model=reasoned_model(Prompt),
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Here are several input/output examples. Create a prompt that could generate all of them:\n\n{examples}",
            },
        ],
    )


async def combine_prompts(data: list[Datum], prompts: list[Prompt]) -> Prompt:
    data = sample(data, k=settings.batch_size)
    return await client.chat.completions.create(
        response_model=reasoned_model(Prompt),
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Here are the prompts to combine:\n\n{prompts}",
            },
            {
                "role": "user",
                "content": f"Here are the examples to generate:\n\n{data}",
            },
            {
                "role": "user",
                "content": "Create a single prompt that could generate all of these examples.",
            },
        ],
    )
