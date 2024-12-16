from hypellm import IO, Example

from .base import dspy, lm


class Questions(dspy.Signature):
    """
    Generate diverse, high-quality questions that can be directly answered by the given text.

    Do's:
        - Generate questions that cover different aspects of the text
        - Include both factual and conceptual questions
        - Make questions clear and unambiguous
        - Ensure questions can be definitively answered by the text
        - Use natural, conversational language

    Don'ts:
        - Don't generate questions about information not present in the text
        - Don't repeat similar questions with minor wording changes
        - Don't use overly complex or technical language
        - Don't make questions too broad or vague

    Reasoning Steps:
        1. Identify the key facts, concepts and details in the text
        2. Consider different question types (who/what/when/where/why/how)
        3. Frame questions to target specific pieces of information
        4. Verify each question is clearly answered in the text
        5. Review and refine question wording for clarity
    """

    outputs: IO = dspy.InputField()
    questions: list[str] = dspy.OutputField()


def questions_sync(datum: Example) -> list[str]:
    with dspy.configure(lm=lm.copy(temperature=0.7)):
        return dspy.ChainOfThought(Questions)(outputs=datum.outputs).questions


questions = dspy.asyncify(questions_sync)
