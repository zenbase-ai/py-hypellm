import instructor
import litellm

from hypellm import settings, DataModel, ReasoningSteps

client = instructor.from_litellm(
    litellm.acompletion,
    model=settings.model,
    api_key=settings.api_key,
    api_version=settings.api_version,
    base_url=settings.base_url,
)


def reasoned_model(result_type: type) -> type[DataModel]:
    class Result(DataModel):
        reasoning_steps: ReasoningSteps
        result: result_type

    return Result
