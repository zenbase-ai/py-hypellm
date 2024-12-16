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
        return self.model_dump()


class Datum(DataModel):
    inputs: IO
    reasoning_steps: Optional[ReasoningSteps] = Field(default=None)
    outputs: IO

    def __init__(
        self,
        inputs: IO,
        reasoning_or_outputs: Union[IO, ReasoningSteps],
        outputs_or_none: Optional[IO] = None,
        **kwargs,
    ):
        if outputs_or_none is None:
            reasoning_steps = None
            outputs = reasoning_or_outputs
        else:
            reasoning_steps = reasoning_or_outputs
            outputs = outputs_or_none

        super().__init__(inputs=inputs, reasoning_steps=reasoning_steps, outputs=outputs, **kwargs)


class Prompt(DataModel):
    intent: str = Field(description="The core purpose or goal of this prompt")
    dos: list[str] = Field(
        description="List of specific instructions the model should follow",
        default_factory=list,
    )
    donts: list[str] = Field(
        description="List of behaviors or outputs the model should avoid",
        default_factory=list,
    )
    reasoning_steps: list[str] = Field(
        description="Step-by-step process the model should use to arrive at the answer",
        default_factory=list,
    )
    examples: list[Datum] = Field(
        description="Sample input-output pairs demonstrating desired behavior",
        default_factory=list,
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
            dos=dos or [],
            donts=donts or [],
            reasoning_steps=reasoning_steps or [],
            examples=examples or [],
            **kwargs,
        )
