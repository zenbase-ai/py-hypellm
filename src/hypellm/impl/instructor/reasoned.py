from functools import partial
from typing import Optional
from random import sample

from hypellm.helpers import amap
from hypellm import IO, Datum, Prompt, ReasoningSteps, DataModel, settings

from .inferred import inferred
from .base import client


async def reasoned(
    data: list[Datum],
    branching_factor: int = 3,
    concurrency: Optional[int] = None,
    prompt: Optional[Prompt] = None,
) -> tuple[Prompt, list[Datum]]:
    """
    Generate hypothetical reasoning steps for a list of input/output examples.

    This function:
    1. Samples a subset of the data
    2. Infers a prompt from the sample
    3. Generates initial reasoning steps for the sample
    4. Uses those examples to generate reasoning steps for the full dataset

    Args:
        data: List of input/output example pairs to generate reasoning for
        batch_size: Number of examples to process at once
        concurrency: Maximum number of parallel operations

    Returns:
        List of reasoning steps for each example in the input data
    """
    assert 1 <= branching_factor <= 8, "branching_factor must be between 1 and 8"

    if concurrency is None:
        concurrency = settings.concurrency

    assert concurrency > 0, "concurrency must be greater than 0"

    # Sample data and infer a prompt
    sample_size = settings.batch_size
    sample_indices = sample(range(len(data)), k=sample_size)
    sample_data = [data[i] for i in sample_indices]

    # Sample prompt is f(inputs) => outputs
    if prompt is None:
        prompt = await inferred(sample_data, settings.batch_size, concurrency)

    # Create a sample of inputs => (thought[], outputs)
    sample_reasonings = await amap(
        partial(infill_reasoning, prompt, branching_factor),
        sample_data,
        batch_size=1,
        concurrency=concurrency,
    )

    # Now generate reasonings for remaining data
    remaining_indices = [i for i in range(len(data)) if i not in sample_indices]
    remaining_data = [data[i] for i in remaining_indices]
    few_shot_prompt = prompt.update(
        examples=[
            Datum(
                inputs=datum.inputs,
                outputs=datum.outputs,
                reasoning=reasoning,
            )
            for datum, reasoning in zip(sample_data, sample_reasonings)
        ]
    )
    remaining_reasonings = await amap(
        partial(infill_reasoning, few_shot_prompt, branching_factor),
        remaining_data,
        batch_size=1,
        concurrency=concurrency,
    )

    # Combine results in original order
    results = [None] * len(data)
    for idx, reasoning in zip(sample_indices, sample_reasonings):
        results[idx] = data[idx].update(reasoning=reasoning)
    for idx, reasoning in zip(remaining_indices, remaining_reasonings):
        results[idx] = data[idx].update(reasoning=reasoning)

    return few_shot_prompt, results


async def infill_reasoning(
    fn_prompt: Prompt, branching_factor: int, datum: Datum
) -> ReasoningSteps:
    user_prompt = Prompt(
        intent="Find the best reasoning trajectory to go from the inputs to the outputs.",
        dos=[
            f"Explore {branching_factor} step by step reasoning trajectories",
            "Reflect on the reasoning trajectories",
            "Select the best reasoning trajectory",
        ],
        donts=[
            "Skip any steps in your reasoning",
        ],
    )
    branches: ThoughtBranches = await client.chat.completions.create(
        response_model=ThoughtBranches,
        messages=[
            {"role": "system", "content": fn_prompt.json()},
            {"role": "user", "content": user_prompt.json()},
            {"role": "user", "content": datum.json()},
        ],
    )
    return branches.best_reasoning


class ReasoningTrajectory(DataModel):
    reasoning: ReasoningSteps
    outputs: IO
    reflection: ReasoningSteps


class ThoughtBranches(DataModel):
    trajectories: list[ReasoningTrajectory]
    best_reasoning: ReasoningSteps
