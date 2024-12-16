import dspy

from hypellm import settings, Datum

lm = dspy.LM(
    model=settings.model,
    api_key=settings.api_key,
    api_version=settings.api_version,
    base_url=settings.base_url,
)

dspy.configure(lm=lm, async_max_workers=settings.concurrency)


def train_dev_split(data: list[Datum], test_size: float = 0.2) -> tuple[list[Datum], list[Datum]]:
    import numpy as np

    np.random.seed(42)
    np.random.shuffle(data)
    return data[: int(len(data) * (1 - test_size))], data[int(len(data) * (1 - test_size)) :]
