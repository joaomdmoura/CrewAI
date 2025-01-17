from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest
from pydantic import BaseModel, Field

from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser


def test_simple_model():
    class SimpleModel(BaseModel):
        field1: int
        field2: str

    parser = PydanticSchemaParser(model=SimpleModel)
    schema = parser.get_schema()

    expected_schema = """{
    field1: int,
    field2: str
}"""
    assert schema.strip() == expected_schema.strip()


def test_nested_model():
    class NestedModel(BaseModel):
        nested_field: int

    class ParentModel(BaseModel):
        parent_field: str
        nested: NestedModel

    parser = PydanticSchemaParser(model=ParentModel)
    schema = parser.get_schema()

    expected_schema = """{
    parent_field: str,
    nested: NestedModel
    {
        nested_field: int
    }
}"""
    assert schema.strip() == expected_schema.strip()


def test_model_with_list():
    class ListModel(BaseModel):
        list_field: List[int]

    parser = PydanticSchemaParser(model=ListModel)
    schema = parser.get_schema()

    expected_schema = """{
    list_field: List[int]
}"""
    assert schema.strip() == expected_schema.strip()


def test_model_with_optional_field():
    class OptionalModel(BaseModel):
        optional_field: Optional[str]

    parser = PydanticSchemaParser(model=OptionalModel)
    schema = parser.get_schema()

    expected_schema = """{
    optional_field: Optional[str]
}"""
    assert schema.strip() == expected_schema.strip()


def test_model_with_union():
    class UnionModel(BaseModel):
        union_field: Union[int, str]

    parser = PydanticSchemaParser(model=UnionModel)
    schema = parser.get_schema()

    expected_schema = """{
    union_field: Union[int, str]
}"""
    assert schema.strip() == expected_schema.strip()


def test_model_with_dict():
    class DictModel(BaseModel):
        dict_field: Dict[str, int]

    parser = PydanticSchemaParser(model=DictModel)
    schema = parser.get_schema()

    expected_schema = """{
    dict_field: Dict[str, int]
}"""
    assert schema.strip() == expected_schema.strip()
