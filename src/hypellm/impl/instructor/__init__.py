from hypellm.helpers import syncify

from .inferred import inferred
from .questions import questions
from .reasoned import reasoned

inferred_sync = syncify(inferred)
questions_sync = syncify(questions)
reasoned_sync = syncify(reasoned)

__all__ = [
    "inferred",
    "questions",
    "reasoned",
    "inferred_sync",
    "questions_sync",
    "reasoned_sync",
]
