from itertools import chain
from pprint import pprint

import pytest

import hypellm

hypellm.settings.impl_name = "instructor"
pytestmark = pytest.mark.impl


@pytest.mark.asyncio
async def test_inferred(medical_classification_dataset):
    prompt = await hypellm.recipes.inferred(medical_classification_dataset)
    assert all(example in medical_classification_dataset for example in prompt.examples)
    pprint(prompt)


@pytest.mark.asyncio
async def test_reasoned(medical_classification_dataset):
    prompt, results = await hypellm.recipes.reasoned(medical_classification_dataset)
    assert all(example in results for example in prompt.examples)
    pprint([prompt, results])


@pytest.mark.asyncio
async def test_questions(medical_classification_dataset):
    questions = await hypellm.recipes.questions(medical_classification_dataset)
    assert set(chain.from_iterable(questions.values())) == set(medical_classification_dataset)
    pprint(questions)


@pytest.mark.asyncio
async def test_inverted(medical_classification_dataset):
    prompt, results = await hypellm.recipes.inverted(medical_classification_dataset)
    assert all(example in results for example in prompt.examples)
    pprint([prompt, results])
