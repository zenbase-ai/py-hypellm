import sys
from typing import Optional, TypeVar, Union
from pydantic import BaseModel, Field

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")
IO = TypeVar("IO", bound=Union[str, dict])
ReasoningSteps = TypeVar("ReasoningSteps", bound=list[str])


class DataModel(BaseModel):
    def update(self, **kwargs) -> "DataModel":
        return self.model_copy(update=kwargs)

    def toDict(self) -> dict:
        return self.model_dump(exclude_none=True)


class Datum(DataModel):
    inputs: IO
    reasoning: Optional[ReasoningSteps] = Field(default=None)
    outputs: IO

    def __init__(
        self,
        inputs: IO,
        reasoning_or_outputs: Union[IO, ReasoningSteps] = None,
        outputs_or_none: Optional[IO] = None,
        *,
        reasoning: Optional[ReasoningSteps] = None,
        outputs: Optional[IO] = None,
        **kwargs,
    ):
        """
        Convenient way to initialize a Datum.

        args:
            Datum(inputs, reasoning, outputs)
            Datum(inputs, outputs)
        kwargs:
            Datum(inputs=..., reasoning=..., outputs=...)
            Datum(inputs=..., outputs=...)
        """
        if reasoning is None and outputs is None:
            if outputs_or_none is None:
                reasoning = None
                outputs = reasoning_or_outputs
            else:
                reasoning = reasoning_or_outputs
                outputs = outputs_or_none

        super().__init__(inputs=inputs, reasoning=reasoning, outputs=outputs, **kwargs)


class Prompt(DataModel):
    intent: str = Field(description="The core purpose or goal of this prompt")
    dos: Optional[list[str]] = Field(
        description="List of specific instructions the model should follow"
    )
    donts: Optional[list[str]] = Field(
        description="List of behaviors or outputs the model should avoid"
    )
    reasoning_steps: Optional[list[str]] = Field(
        description="Step-by-step process the model should use to arrive at the answer"
    )
    examples: Optional[list[Datum]] = Field(
        description="Sample input-output pairs demonstrating desired behavior"
    )

    def __init__(
        self,
        intent: str,
        *,
        dos: Optional[list[str]] = None,
        donts: Optional[list[str]] = None,
        reasoning_steps: Optional[list[str]] = None,
        examples: Optional[list[Datum]] = None,
        **kwargs,
    ):
        super().__init__(
            intent=intent,
            dos=dos,
            donts=donts,
            reasoning_steps=reasoning_steps,
            examples=examples,
            **kwargs,
        )
