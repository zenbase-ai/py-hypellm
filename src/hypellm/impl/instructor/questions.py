from hypellm.types import Datum, Prompt, DataModel, ReasoningSteps

from .base import client


async def questions(datum: Datum) -> list[str]:
    response: HypotheticalQuestions = await client.chat.completions.create(
        response_model=HypotheticalQuestions,
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.json()},
            {"role": "user", "content": datum.json()},
        ],
    )
    return response.questions


class HypotheticalQuestions(DataModel):
    reasoning_steps: ReasoningSteps
    questions: list[str]


SYSTEM_PROMPT = Prompt(
    intent="Generate diverse, high-quality questions that can be directly answered by the given text.",
    dos=[
        "Generate questions that cover different aspects of the text",
        "Include both factual and conceptual questions",
        "Make questions clear and unambiguous",
        "Ensure questions can be definitively answered by the text",
        "Use natural, conversational language",
    ],
    donts=[
        "Don't generate questions about information not present in the text",
        "Don't repeat similar questions with minor wording changes",
        "Don't use overly complex or technical language",
        "Don't make questions too broad or vague",
    ],
    reasoning_steps=[
        "1. Identify the key facts, concepts and details in the text",
        "2. Consider different question types (who/what/when/where/why/how)",
        "3. Frame questions to target specific pieces of information",
        "4. Verify each question is clearly answered in the text",
        "5. Review and refine question wording for clarity",
    ],
)
