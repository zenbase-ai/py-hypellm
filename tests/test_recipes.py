import pytest

import hypellm

hypellm.settings.impl_name = "helpers.mock_impl"
hypellm.settings.show_progress = False


@pytest.fixture
def mock_data():
    return [
        hypellm.Result(inputs="What is 2+2?", outputs="4"),
        hypellm.Result(inputs="What is the capital of France?", outputs="Paris"),
    ]


@pytest.mark.asyncio
async def test_inferred(mock_data):
    prompt = await hypellm.recipes.inferred(mock_data)
    assert prompt.intent == "inferred intent"
    assert prompt.dos == ["do 1", "do 2"]
    assert prompt.donts == ["dont 1", "dont 2"]
    assert prompt.reasoning_steps == ["step 1", "step 2"]
    assert prompt.examples == mock_data[:2]


@pytest.mark.asyncio
async def test_reasoned(mock_data):
    prompt, results = await hypellm.recipes.reasoned(mock_data)
    assert len(results) == len(mock_data)
    assert all(example in results for example in prompt.examples)
    assert all(isinstance(r, hypellm.Result) for r in results)
    assert all(hasattr(r, "reasoning") for r in results)
    assert all(r.reasoning == ["step 1", "step 2", "step 3"] for r in results)


def test_reasoned_sync(mock_data):
    prompt, results = hypellm.recipes.reasoned_sync(mock_data)
    assert len(results) == len(mock_data)
    assert all(example in results for example in prompt.examples)
    assert all(isinstance(r, hypellm.Result) for r in results)
    assert all(hasattr(r, "reasoning") for r in results)
    assert all(r.reasoning == ["step 1", "step 2", "step 3"] for r in results)


@pytest.mark.asyncio
async def test_inverted(mock_data):
    prompt, results = await hypellm.recipes.inverted(mock_data)
    assert prompt.intent == "inferred intent"
    assert all(example in results for example in prompt.examples)
    assert len(results) == len(mock_data)
    assert all(
        r.inputs == mock.outputs and r.outputs == mock.inputs for r, mock in zip(results, mock_data)
    )


def test_inverted_sync(mock_data):
    prompt, results = hypellm.recipes.inverted_sync(mock_data)
    assert prompt.intent == "inferred intent"
    assert all(example in results for example in prompt.examples)
    assert len(results) == len(mock_data)
    assert all(
        r.inputs == mock.outputs and r.outputs == mock.inputs for r, mock in zip(results, mock_data)
    )


@pytest.mark.asyncio
async def test_questions(mock_data):
    results = await hypellm.recipes.questions(mock_data)
    assert isinstance(results, dict)
    assert all(len(questions) == len(mock_data) for questions in results.values())


def test_questions_sync(mock_data):
    results = hypellm.recipes.questions_sync(mock_data)
    assert isinstance(results, dict)
    assert all(len(questions) == len(mock_data) for questions in results.values())
