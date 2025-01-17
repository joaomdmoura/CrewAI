import json
import re
from typing import Any, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from crewai.agents.agent_builder.utilities.base_output_converter import OutputConverter
from crewai.utilities.printer import Printer
from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser


class ConverterError(Exception):
    """Error raised when Converter fails to parse the input."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class Converter(OutputConverter):
    """Class that converts text into either pydantic or json."""

    def to_pydantic(self, current_attempt=1):
        """Convert text to pydantic."""
        try:
            if self.llm.supports_function_calling():
                return self._create_instructor().to_pydantic()
            else:
                response = self.llm.call(
                    [
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": self.text},
                    ]
                )
                return self.model.model_validate_json(response)
        except ValidationError as e:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to the following validation error: {e}"
            )
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to the following error: {e}"
            )

    def to_json(self, current_attempt=1):
        """Convert text to json."""
        try:
            if self.llm.supports_function_calling():
                return self._create_instructor().to_json()
            else:
                return json.dumps(
                    self.llm.call(
                        [
                            {"role": "system", "content": self.instructions},
                            {"role": "user", "content": self.text},
                        ]
                    )
                )
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_json(current_attempt + 1)
            return ConverterError(f"Failed to convert text into JSON, error: {e}.")

    def _create_instructor(self):
        """Create an instructor."""
        from crewai.utilities import InternalInstructor

        inst = InternalInstructor(
            llm=self.llm,
            model=self.model,
            content=self.text,
        )
        return inst

    def _convert_with_instructions(self):
        """Create a chain."""
        from crewai.utilities.crew_pydantic_output_parser import (
            CrewPydanticOutputParser,
        )

        parser = CrewPydanticOutputParser(pydantic_object=self.model)
        result = self.llm.call(
            [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": self.text},
            ]
        )
        return parser.parse_result(result)


def convert_to_model(
    result: str,
    output_pydantic: Optional[Type[BaseModel]],
    output_json: Optional[Type[BaseModel]],
    agent: Any,
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    model = output_pydantic or output_json
    if model is None:
        return result
    try:
        escaped_result = json.dumps(json.loads(result, strict=False))
        return validate_model(escaped_result, model, bool(output_json))
    except json.JSONDecodeError:
        return handle_partial_json(
            result, model, bool(output_json), agent, converter_cls
        )

    except ValidationError:
        return handle_partial_json(
            result, model, bool(output_json), agent, converter_cls
        )

    except Exception as e:
        Printer().print(
            content=f"Unexpected error during model conversion: {type(e).__name__}: {e}. Returning original result.",
            color="red",
        )
        return result


def validate_model(
    result: str, model: Type[BaseModel], is_json_output: bool
) -> Union[dict, BaseModel]:
    exported_result = model.model_validate_json(result)
    if is_json_output:
        return exported_result.model_dump()
    return exported_result


def handle_partial_json(
    result: str,
    model: Type[BaseModel],
    is_json_output: bool,
    agent: Any,
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    match = re.search(r"({.*})", result, re.DOTALL)
    if match:
        try:
            exported_result = model.model_validate_json(match.group(0))
            if is_json_output:
                return exported_result.model_dump()
            return exported_result
        except json.JSONDecodeError:
            pass
        except ValidationError:
            pass
        except Exception as e:
            Printer().print(
                content=f"Unexpected error during partial JSON handling: {type(e).__name__}: {e}. Attempting alternative conversion method.",
                color="red",
            )

    return convert_with_instructions(
        result, model, is_json_output, agent, converter_cls
    )


def convert_with_instructions(
    result: str,
    model: Type[BaseModel],
    is_json_output: bool,
    agent: Any,
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    llm = agent.function_calling_llm or agent.llm
    instructions = get_conversion_instructions(model, llm)
    converter = create_converter(
        agent=agent,
        converter_cls=converter_cls,
        llm=llm,
        text=result,
        model=model,
        instructions=instructions,
    )
    exported_result = (
        converter.to_pydantic() if not is_json_output else converter.to_json()
    )

    if isinstance(exported_result, ConverterError):
        Printer().print(
            content=f"{exported_result.message} Using raw output instead.",
            color="red",
        )
        return result

    return exported_result


def get_conversion_instructions(model: Type[BaseModel], llm: Any) -> str:
    instructions = "Please convert the following text into valid JSON."
    if llm.supports_function_calling():
        model_schema = PydanticSchemaParser(model=model).get_schema()
        instructions += (
            f"\n\nThe JSON should follow this schema:\n```json\n{model_schema}\n```"
        )
    else:
        model_description = generate_model_description(model)
        instructions += f"\n\nThe JSON should follow this format:\n{model_description}"
    return instructions


def create_converter(
    agent: Optional[Any] = None,
    converter_cls: Optional[Type[Converter]] = None,
    *args,
    **kwargs,
) -> Converter:
    if agent and not converter_cls:
        if hasattr(agent, "get_output_converter"):
            converter = agent.get_output_converter(*args, **kwargs)
        else:
            raise AttributeError("Agent does not have a 'get_output_converter' method")
    elif converter_cls:
        converter = converter_cls(*args, **kwargs)
    else:
        raise ValueError("Either agent or converter_cls must be provided")

    if not converter:
        raise Exception("No output converter found or set.")

    return converter


def generate_model_description(model: Type[BaseModel]) -> str:
    """
    Generate a string description of a Pydantic model's fields and their types.

    This function takes a Pydantic model class and returns a string that describes
    the model's fields and their respective types. The description includes handling
    of complex types such as `Optional`, `List`, and `Dict`, as well as nested Pydantic
    models.
    """

    def describe_field(field_type):
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union or (origin is None and len(args) > 0):
            # Handle both Union and the new '|' syntax
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return f"Optional[{describe_field(non_none_args[0])}]"
            else:
                return f"Optional[Union[{', '.join(describe_field(arg) for arg in non_none_args)}]]"
        elif origin is list:
            return f"List[{describe_field(args[0])}]"
        elif origin is dict:
            key_type = describe_field(args[0])
            value_type = describe_field(args[1])
            return f"Dict[{key_type}, {value_type}]"
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return generate_model_description(field_type)
        elif hasattr(field_type, "__name__"):
            return field_type.__name__
        else:
            return str(field_type)

    fields = model.__annotations__
    field_descriptions = [
        f'"{name}": {describe_field(type_)}' for name, type_ in fields.items()
    ]
    return "{\n  " + ",\n  ".join(field_descriptions) + "\n}"
