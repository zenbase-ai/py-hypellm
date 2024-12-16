from functools import partial
from random import sample
from typing import Optional

from hypellm import Datum, settings, ReasoningSteps, IO, DataModel, Prompt
from hypellm.helpers import pmap
from .base import dspy
from .inferred import inferred_sync


class ReasoningTrajectory(DataModel):
    reasoning: ReasoningSteps
    outputs: IO
    reflection: ReasoningSteps


class ThoughtBranches(DataModel):
    trajectories: list[ReasoningTrajectory]
    best_reasoning: ReasoningSteps


class Think(dspy.Signature):
    prompt: Prompt = dspy.InputField()
    inputs: IO = dspy.InputField()
    outputs: IO = dspy.InputField()
    branching_factor: int = dspy.InputField()
    branches: ThoughtBranches = dspy.OutputField()


def infill_reasoning(fn_prompt: Prompt, branching_factor: int, datum: Datum) -> ReasoningSteps:
    branches: ThoughtBranches = dspy.Predict(Think)(
        prompt=fn_prompt,
        branching_factor=branching_factor,
        inputs=datum.inputs,
        outputs=datum.outputs,
    )
    return branches.best_reasoning


def reasoned_sync(
    data: list[Datum],
    branching_factor: int = 3,
    concurrency: Optional[int] = None,
) -> list[Datum]:
    concurrency = concurrency or settings.concurrency

    sample_data = sample(data, k=settings.batch_size)
    sample_prompt = inferred_sync(sample_data, settings.batch_size, concurrency)

    sample_reasonings = pmap(
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

    few_shot_prompt = sample_prompt.update(examples=sample_examples)

    data_reasonings = pmap(
        partial(infill_reasoning, few_shot_prompt, branching_factor),
        data,
        batch_size=1,
        concurrency=concurrency,
    )

    return [datum.update(reasoning=reasoning) for datum, reasoning in zip(data, data_reasonings)]


reasoned = dspy.asyncify(reasoned_sync)
