from itertools import chain
from pprint import pprint
import pytest

import hypellm

hypellm.settings.impl_name = "instructor"

pytestmark = pytest.mark.impl


@pytest.fixture
def mock_data():
    return [
        hypellm.Datum(
            inputs="The patient presents with elevated troponin levels (0.8 ng/mL) and ST-segment depression, but no chest pain or dyspnea.",
            outputs="unstable_angina",
        ),
        hypellm.Datum(
            inputs="Labs show WBC 15k/μL with 80% neutrophils, fever 39.2°C, and consolidation in right lower lobe on chest X-ray.",
            outputs="bacterial_pneumonia",
        ),
        hypellm.Datum(
            inputs="Sudden onset vertigo with horizontal nystagmus, normal head CT, negative Dix-Hallpike, no hearing loss.",
            outputs="vestibular_neuritis",
        ),
        hypellm.Datum(
            inputs="Progressive weakness in lower extremities, decreased DTRs, EMG shows demyelination pattern, CSF protein elevated.",
            outputs="guillain_barre",
        ),
        hypellm.Datum(
            inputs="Recurrent episodes of focal seizures with preserved awareness, MRI shows temporal lobe calcification.",
            outputs="mesial_temporal_sclerosis",
        ),
    ]


@pytest.mark.asyncio
async def test_inferred(mock_data):
    prompt = await hypellm.recipes.inferred(mock_data)
    assert all(example in mock_data for example in prompt.examples)
    pprint(prompt)


@pytest.mark.asyncio
async def test_reasoned(mock_data):
    prompt, results = await hypellm.recipes.reasoned(mock_data)
    assert all(example in results for example in prompt.examples)
    pprint([prompt, results])


@pytest.mark.asyncio
async def test_questions(mock_data):
    questions = await hypellm.recipes.questions(mock_data)
    assert set(chain.from_iterable(questions.values())) == set(mock_data)
    pprint(questions)


@pytest.mark.asyncio
async def test_inverted(mock_data):
    prompt, results = await hypellm.recipes.inverted(mock_data)
    assert all(example in results for example in prompt.examples)
    pprint([prompt, results])
