from functools import partial
from random import sample
from typing import Optional

import ujson

from hypellm import settings, DataModel, Prompt, ReasoningSteps, Datum
from hypellm.helpers import amap

from .base import client


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
    candidates = await amap(infer_prompt, data, batch_size, concurrency)
    while len(candidates) > 1:
        candidates = await amap(
            partial(combine_prompts, data),
            candidates,
            batch_size,
            concurrency,
        )

    return candidates[0]


async def infer_prompt(examples: list[Datum]) -> Prompt:
    response: HypotheticalPrompt = await client.chat.completions.create(
        response_model=HypotheticalPrompt,
        temperature=0.42,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.json(),
            },
            {
                "role": "user",
                "content": f"Here are several input/output examples. Create a prompt that could generate all of them:\n\n{ujson.dumps(examples)}",
            },
        ],
    )
    return response.prompt


async def combine_prompts(data: list[Datum], prompts: list[Prompt]) -> Prompt:
    data = sample(data, k=settings.batch_size)
    response: HypotheticalPrompt = await client.chat.completions.create(
        response_model=HypotheticalPrompt,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.json(),
            },
            {
                "role": "user",
                "content": f"Here are the prompts to combine:\n\n{ujson.dumps(prompts)}",
            },
            {
                "role": "user",
                "content": f"Here are the examples to generate:\n\n{ujson.dumps(data)}",
            },
            {
                "role": "user",
                "content": "Create a single prompt that could generate all of these examples.",
            },
        ],
    )
    return response.prompt


class HypotheticalPrompt(DataModel):
    reasoning_steps: ReasoningSteps
    prompt: Prompt


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
