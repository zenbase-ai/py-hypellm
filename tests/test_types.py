from hypothesis import given
from hypothesis.strategies import dictionaries, text

from hypellm.types import dump_io_value, is_io_value


@given(value=text())
def test_str_is_io_value(value: str):
    assert is_io_value(value)


@given(value=dictionaries(text(), text()))
def test_dict_is_io_value(value: dict[str, str]):
    assert is_io_value(value)


def test_data_key():
    data = {"b": 2, "a": 1}
    assert dump_io_value(data) == '{"a":1,"b":2}'
