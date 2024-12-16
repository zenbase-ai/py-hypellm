from functools import partial
from typing import Optional
from random import shuffle, sample

from hypellm.helpers import amap
from hypellm import IO, Datum, Prompt, ReasoningSteps, DataModel, settings

from .inferred import inferred
from .base import client


async def reasoned(
    data: list[Datum],
    branching_factor: int = 3,
    concurrency: Optional[int] = None,
) -> list[Datum]:
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

    concurrency = concurrency or settings.concurrency

    # Sample data and infer a prompt
    sample_data = sample(data, k=settings.batch_size)
    # Sample prompt is f(inputs) => outputs
    sample_prompt = await inferred(sample_data, settings.batch_size, concurrency)
    # Create a sample of inputs => (thought[], outputs)
    sample_reasonings = await amap(
        partial(infill_reasoning, sample_prompt, branching_factor),
        sample_data,
        batch_size=1,
        concurrency=concurrency,
    )
    sample_examples = [
        Datum(
            inputs=datum.inputs,
            outputs=datum.outputs,
            reasoning_steps=reasoning,
        )
        for datum, reasoning in zip(sample_data, sample_reasonings)
    ]
    shuffle(sample_examples)

    # Now generate reasonings for the entire dataset
    few_shot_prompt = sample_prompt.update(examples=sample_examples)
    data_reasonings = await amap(
        partial(infill_reasoning, few_shot_prompt, branching_factor),
        data,
        batch_size=1,
        concurrency=concurrency,
    )

    return [datum.update(reasoning=reasoning) for datum, reasoning in zip(data, data_reasonings)]


class ReasoningTrajectory(DataModel):
    reasoning: ReasoningSteps
    outputs: IO
    reflection: ReasoningSteps


class ThoughtBranches(DataModel):
    trajectories: list[ReasoningTrajectory]
    best_reasoning: ReasoningSteps


async def infill_reasoning(
    fn_prompt: Prompt, branching_factor: int, datum: Datum
) -> ReasoningSteps:
    branches: ThoughtBranches = await client.chat.completions.create(
        response_model=ThoughtBranches,
        messages=[
            {"role": "system", "content": fn_prompt},
            {
                "role": "user",
                "content": Prompt(
                    intent="Find the best reasoning trajectory to go from the inputs to the outputs.",
                    dos=[
                        f"Explore {branching_factor} step by step reasoning trajectories",
                        "Reflect on the reasoning trajectories",
                    ],
                    donts=[
                        "Skip any steps in your reasoning",
                    ],
                ),
            },
            {"role": "user", "content": datum},
        ],
    )
    return branches.best_reasoning
