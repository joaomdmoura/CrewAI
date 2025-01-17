import json
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from crewai.llm import LLM
from crewai.utilities.converter import (
    Converter,
    ConverterError,
    convert_to_model,
    convert_with_instructions,
    create_converter,
    generate_model_description,
    get_conversion_instructions,
    handle_partial_json,
    validate_model,
)
from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser


# Sample Pydantic models for testing
class EmailResponse(BaseModel):
    previous_message_content: str


class EmailResponses(BaseModel):
    responses: list[EmailResponse]


class SimpleModel(BaseModel):
    name: str
    age: int


class NestedModel(BaseModel):
    id: int
    data: SimpleModel


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class Person(BaseModel):
    name: str
    age: int
    address: Address


class CustomConverter(Converter):
    pass


# Fixtures
@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.function_calling_llm = None
    agent.llm = Mock()
    return agent


# Tests for convert_to_model
def test_convert_to_model_with_valid_json():
    result = '{"name": "John", "age": 30}'
    output = convert_to_model(result, SimpleModel, None, None)
    assert isinstance(output, SimpleModel)
    assert output.name == "John"
    assert output.age == 30


def test_convert_to_model_with_invalid_json():
    result = '{"name": "John", "age": "thirty"}'
    with patch("crewai.utilities.converter.handle_partial_json") as mock_handle:
        mock_handle.return_value = "Fallback result"
        output = convert_to_model(result, SimpleModel, None, None)
        assert output == "Fallback result"


def test_convert_to_model_with_no_model():
    result = "Plain text"
    output = convert_to_model(result, None, None, None)
    assert output == "Plain text"


def test_convert_to_model_with_special_characters():
    json_string_test = """
    {
        "responses": [
            {
                "previous_message_content": "Hi Tom,\r\n\r\nNiamh has chosen the Mika phonics on"
            }
        ]
    }
    """
    output = convert_to_model(json_string_test, EmailResponses, None, None)
    assert isinstance(output, EmailResponses)
    assert len(output.responses) == 1
    assert (
        output.responses[0].previous_message_content
        == "Hi Tom,\r\n\r\nNiamh has chosen the Mika phonics on"
    )


def test_convert_to_model_with_escaped_special_characters():
    json_string_test = json.dumps(
        {
            "responses": [
                {
                    "previous_message_content": "Hi Tom,\r\n\r\nNiamh has chosen the Mika phonics on"
                }
            ]
        }
    )
    output = convert_to_model(json_string_test, EmailResponses, None, None)
    assert isinstance(output, EmailResponses)
    assert len(output.responses) == 1
    assert (
        output.responses[0].previous_message_content
        == "Hi Tom,\r\n\r\nNiamh has chosen the Mika phonics on"
    )


def test_convert_to_model_with_multiple_special_characters():
    json_string_test = """
    {
        "responses": [
            {
                "previous_message_content": "Line 1\r\nLine 2\tTabbed\nLine 3\r\n\rEscaped newline"
            }
        ]
    }
    """
    output = convert_to_model(json_string_test, EmailResponses, None, None)
    assert isinstance(output, EmailResponses)
    assert len(output.responses) == 1
    assert (
        output.responses[0].previous_message_content
        == "Line 1\r\nLine 2\tTabbed\nLine 3\r\n\rEscaped newline"
    )


# Tests for validate_model
def test_validate_model_pydantic_output():
    result = '{"name": "Alice", "age": 25}'
    output = validate_model(result, SimpleModel, False)
    assert isinstance(output, SimpleModel)
    assert output.name == "Alice"
    assert output.age == 25


def test_validate_model_json_output():
    result = '{"name": "Bob", "age": 40}'
    output = validate_model(result, SimpleModel, True)
    assert isinstance(output, dict)
    assert output == {"name": "Bob", "age": 40}


# Tests for handle_partial_json
def test_handle_partial_json_with_valid_partial():
    result = 'Some text {"name": "Charlie", "age": 35} more text'
    output = handle_partial_json(result, SimpleModel, False, None)
    assert isinstance(output, SimpleModel)
    assert output.name == "Charlie"
    assert output.age == 35


def test_handle_partial_json_with_invalid_partial(mock_agent):
    result = "No valid JSON here"
    with patch("crewai.utilities.converter.convert_with_instructions") as mock_convert:
        mock_convert.return_value = "Converted result"
        output = handle_partial_json(result, SimpleModel, False, mock_agent)
        assert output == "Converted result"


# Tests for convert_with_instructions
@patch("crewai.utilities.converter.create_converter")
@patch("crewai.utilities.converter.get_conversion_instructions")
def test_convert_with_instructions_success(
    mock_get_instructions, mock_create_converter, mock_agent
):
    mock_get_instructions.return_value = "Instructions"
    mock_converter = Mock()
    mock_converter.to_pydantic.return_value = SimpleModel(name="David", age=50)
    mock_create_converter.return_value = mock_converter

    result = "Some text to convert"
    output = convert_with_instructions(result, SimpleModel, False, mock_agent)

    assert isinstance(output, SimpleModel)
    assert output.name == "David"
    assert output.age == 50


@patch("crewai.utilities.converter.create_converter")
@patch("crewai.utilities.converter.get_conversion_instructions")
def test_convert_with_instructions_failure(
    mock_get_instructions, mock_create_converter, mock_agent
):
    mock_get_instructions.return_value = "Instructions"
    mock_converter = Mock()
    mock_converter.to_pydantic.return_value = ConverterError("Conversion failed")
    mock_create_converter.return_value = mock_converter

    result = "Some text to convert"
    with patch("crewai.utilities.converter.Printer") as mock_printer:
        output = convert_with_instructions(result, SimpleModel, False, mock_agent)
        assert output == result
        mock_printer.return_value.print.assert_called_once()


# Tests for get_conversion_instructions
def test_get_conversion_instructions_gpt():
    llm = LLM(model="gpt-4o-mini")
    with patch.object(LLM, "supports_function_calling") as supports_function_calling:
        supports_function_calling.return_value = True
        instructions = get_conversion_instructions(SimpleModel, llm)
        model_schema = PydanticSchemaParser(model=SimpleModel).get_schema()
        assert (
            instructions
            == f"Please convert the following text into valid JSON.\n\nThe JSON should follow this schema:\n```json\n{model_schema}\n```"
        )


def test_get_conversion_instructions_non_gpt():
    llm = LLM(model="ollama/llama3.1", base_url="http://localhost:11434")
    with patch.object(LLM, "supports_function_calling", return_value=False):
        instructions = get_conversion_instructions(SimpleModel, llm)
        assert '"name": str' in instructions
        assert '"age": int' in instructions


# Tests for is_gpt
def test_supports_function_calling_true():
    llm = LLM(model="gpt-4o")
    assert llm.supports_function_calling() is True


def test_supports_function_calling_false():
    llm = LLM(model="non-existent-model")
    assert llm.supports_function_calling() is False


def test_create_converter_with_mock_agent():
    mock_agent = MagicMock()
    mock_agent.get_output_converter.return_value = MagicMock(spec=Converter)

    converter = create_converter(
        agent=mock_agent,
        llm=Mock(),
        text="Sample",
        model=SimpleModel,
        instructions="Convert",
    )

    assert isinstance(converter, Converter)
    mock_agent.get_output_converter.assert_called_once()


def test_create_converter_with_custom_converter():
    converter = create_converter(
        converter_cls=CustomConverter,
        llm=LLM(model="gpt-4o-mini"),
        text="Sample",
        model=SimpleModel,
        instructions="Convert",
    )

    assert isinstance(converter, CustomConverter)


def test_create_converter_fails_without_agent_or_converter_cls():
    with pytest.raises(
        ValueError, match="Either agent or converter_cls must be provided"
    ):
        create_converter(
            llm=Mock(), text="Sample", model=SimpleModel, instructions="Convert"
        )


def test_generate_model_description_simple_model():
    description = generate_model_description(SimpleModel)
    expected_description = '{\n  "name": str,\n  "age": int\n}'
    assert description == expected_description


def test_generate_model_description_nested_model():
    description = generate_model_description(NestedModel)
    expected_description = (
        '{\n  "id": int,\n  "data": {\n  "name": str,\n  "age": int\n}\n}'
    )
    assert description == expected_description


def test_generate_model_description_optional_field():
    class ModelWithOptionalField(BaseModel):
        name: Optional[str]
        age: int

    description = generate_model_description(ModelWithOptionalField)
    expected_description = '{\n  "name": Optional[str],\n  "age": int\n}'
    assert description == expected_description


def test_generate_model_description_list_field():
    class ModelWithListField(BaseModel):
        items: List[int]

    description = generate_model_description(ModelWithListField)
    expected_description = '{\n  "items": List[int]\n}'
    assert description == expected_description


def test_generate_model_description_dict_field():
    class ModelWithDictField(BaseModel):
        attributes: Dict[str, int]

    description = generate_model_description(ModelWithDictField)
    expected_description = '{\n  "attributes": Dict[str, int]\n}'
    assert description == expected_description


@pytest.mark.vcr(filter_headers=["authorization"])
def test_convert_with_instructions():
    llm = LLM(model="gpt-4o-mini")
    sample_text = "Name: Alice, Age: 30"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )

    # Act
    output = converter.to_pydantic()

    # Assert
    assert isinstance(output, SimpleModel)
    assert output.name == "Alice"
    assert output.age == 30


@pytest.mark.vcr(filter_headers=["authorization"])
def test_converter_with_llama3_2_model():
    llm = LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")

    sample_text = "Name: Alice Llama, Age: 30"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, SimpleModel)
    assert output.name == "Alice Llama"
    assert output.age == 30


@pytest.mark.vcr(filter_headers=["authorization"])
def test_converter_with_llama3_1_model():
    llm = LLM(model="ollama/llama3.1", base_url="http://localhost:11434")
    sample_text = "Name: Alice Llama, Age: 30"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, SimpleModel)
    assert output.name == "Alice Llama"
    assert output.age == 30


@pytest.mark.vcr(filter_headers=["authorization"])
def test_converter_with_nested_model():
    llm = LLM(model="gpt-4o-mini")
    sample_text = "Name: John Doe\nAge: 30\nAddress: 123 Main St, Anytown, 12345"

    instructions = get_conversion_instructions(Person, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=Person,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, Person)
    assert output.name == "John Doe"
    assert output.age == 30
    assert isinstance(output.address, Address)
    assert output.address.street == "123 Main St"
    assert output.address.city == "Anytown"
    assert output.address.zip_code == "12345"


# Tests for error handling
def test_converter_error_handling():
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.return_value = "Invalid JSON"
    sample_text = "Name: Alice, Age: 30"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )

    with pytest.raises(ConverterError) as exc_info:
        output = converter.to_pydantic()

    assert "Failed to convert text into a Pydantic model" in str(exc_info.value)


# Tests for retry logic
def test_converter_retry_logic():
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.side_effect = [
        "Invalid JSON",
        "Still invalid",
        '{"name": "Retry Alice", "age": 30}',
    ]
    sample_text = "Name: Retry Alice, Age: 30"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
        max_attempts=3,
    )

    output = converter.to_pydantic()

    assert isinstance(output, SimpleModel)
    assert output.name == "Retry Alice"
    assert output.age == 30
    assert llm.call.call_count == 3


# Tests for optional fields
def test_converter_with_optional_fields():
    class OptionalModel(BaseModel):
        name: str
        age: Optional[int]

    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    # Simulate the LLM's response with 'age' explicitly set to null
    llm.call.return_value = '{"name": "Bob", "age": null}'
    sample_text = "Name: Bob, age: None"

    instructions = get_conversion_instructions(OptionalModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=OptionalModel,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, OptionalModel)
    assert output.name == "Bob"
    assert output.age is None


# Tests for list fields
def test_converter_with_list_field():
    class ListModel(BaseModel):
        items: List[int]

    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.return_value = '{"items": [1, 2, 3]}'
    sample_text = "Items: 1, 2, 3"

    instructions = get_conversion_instructions(ListModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=ListModel,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, ListModel)
    assert output.items == [1, 2, 3]


# Tests for enums
from enum import Enum


def test_converter_with_enum():
    class Color(Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    class EnumModel(BaseModel):
        name: str
        color: Color

    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.return_value = '{"name": "Alice", "color": "red"}'
    sample_text = "Name: Alice, Color: Red"

    instructions = get_conversion_instructions(EnumModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=EnumModel,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, EnumModel)
    assert output.name == "Alice"
    assert output.color == Color.RED


# Tests for ambiguous input
def test_converter_with_ambiguous_input():
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.return_value = '{"name": "Charlie", "age": "Not an age"}'
    sample_text = "Charlie is thirty years old"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )

    with pytest.raises(ConverterError) as exc_info:
        output = converter.to_pydantic()

    assert "validation error" in str(exc_info.value).lower()


# Tests for function calling support
def test_converter_with_function_calling():
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = True

    instructor = Mock()
    instructor.to_pydantic.return_value = SimpleModel(name="Eve", age=35)

    converter = Converter(
        llm=llm,
        text="Name: Eve, Age: 35",
        model=SimpleModel,
        instructions="Convert this text.",
    )
    converter._create_instructor = Mock(return_value=instructor)

    output = converter.to_pydantic()

    assert isinstance(output, SimpleModel)
    assert output.name == "Eve"
    assert output.age == 35
    instructor.to_pydantic.assert_called_once()


def test_generate_model_description_union_field():
    class UnionModel(BaseModel):
        field: int | str | None

    description = generate_model_description(UnionModel)
    expected_description = '{\n  "field": int | str | None\n}'
    assert description == expected_description
