import instructor
import litellm

from hypellm import settings

client = instructor.from_litellm(
    litellm.acompletion,
    model=settings.model,
    api_key=settings.api_key,
    api_version=settings.api_version,
    base_url=settings.base_url,
)
