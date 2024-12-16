import pytest

import ujson

from hypellm.types import Datum, Prompt


def test_datum_initialization():
    # Test with just inputs and outputs
    datum = Datum(inputs="test input", outputs="test output")
    assert datum.inputs == "test input"
    assert datum.outputs == "test output"
    assert datum.reasoning is None

    # Test with reasoning steps
    datum = Datum(
        inputs="test input",
        reasoning_or_outputs=["step 1", "step 2"],
        outputs_or_none="test output",
    )
    assert datum.inputs == "test input"
    assert datum.reasoning == ["step 1", "step 2"]
    assert datum.outputs == "test output"

    # Test with dict inputs/outputs
    datum = Datum(inputs={"key": "input"}, outputs={"key": "output"})
    assert datum.inputs == {"key": "input"}
    assert datum.outputs == {"key": "output"}


def test_datum_methods():
    datum = Datum(inputs="test", outputs="test")

    # Test update method
    updated = datum.update(inputs="new input")
    assert updated.inputs == "new input"
    assert updated.outputs == "test"

    # Test serialization
    assert ujson.dumps(datum) == '{"inputs":"test","outputs":"test"}'


def test_prompt_initialization():
    # Test minimal initialization
    prompt = Prompt(intent="test intent")
    assert prompt.intent == "test intent"
    assert prompt.dos is None
    assert prompt.donts is None
    assert prompt.reasoning_steps is None
    assert prompt.examples is None

    # Test full initialization
    examples = [Datum(inputs="test", outputs="test")]
    prompt = Prompt(
        intent="test intent",
        dos=["do this"],
        donts=["don't do this"],
        reasoning_steps=["step 1"],
        examples=examples,
    )
    assert prompt.intent == "test intent"
    assert prompt.dos == ["do this"]
    assert prompt.donts == ["don't do this"]
    assert prompt.reasoning_steps == ["step 1"]
    assert prompt.examples == examples


def test_prompt_methods():
    prompt = Prompt(intent="test")

    # Test update method
    updated = prompt.update(intent="new intent")
    assert updated.intent == "new intent"

    # Test serialization
    assert ujson.dumps(prompt) == '{"intent":"test"}'


def test_invalid_types():
    # Test invalid input type
    with pytest.raises(ValueError):
        Datum(inputs=123, outputs="test")  # type: ignore

    # Test invalid output type
    with pytest.raises(ValueError):
        Datum(inputs="test", outputs=123)  # type: ignore

    # Test invalid reasoning steps type
    with pytest.raises(ValueError):
        Datum(
            inputs="test",
            reasoning_or_outputs="not a list",  # type: ignore
            outputs_or_none="test",
        )
