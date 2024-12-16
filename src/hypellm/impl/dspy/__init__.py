from .base import dspy

from .reasoned import reasoned_sync
from .questions import questions_sync
from .inferred import inferred_sync

reasoned = dspy.asyncify(reasoned_sync)
questions = dspy.asyncify(questions_sync)
inferred = dspy.asyncify(inferred_sync)
