from typing import Optional

from hypellm import Prompt, Datum, settings, ReasoningSteps, IO

from .base import dspy, train_dev_split


class Function(dspy.Signature):
    inputs: IO = dspy.InputField()
    reasoning_steps: ReasoningSteps = dspy.OutputField()
    outputs: IO = dspy.OutputField()


def inferred_sync(
    data: list[Datum],
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Prompt:
    optimizer = dspy.MIPROv2(
        metric=dspy.evaluate.semantic_f1,
        mode="light",
    )
    trainset, devset = train_dev_split(data)
    best_fn = optimizer.compile(Function, trainset=trainset, devset=devset)

    return Prompt(
        intent=...,
        examples=[
            Datum(
                inputs=example.inputs,
                reasoning_steps=example.reasoning_steps,
                outputs=example.outputs,
            )
            for example in best_fn.demos
        ],
    )


inferred = dspy.asyncify(inferred_sync)
