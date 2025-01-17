import pytest

from crewai.utilities.i18n import I18N


def test_load_prompts():
    i18n = I18N()
    i18n.load_prompts()
    assert i18n._prompts is not None


def test_slice():
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.slice("role_playing"), str)


def test_tools():
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.tools("ask_question"), str)


def test_retrieve():
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)


def test_retrieve_not_found():
    i18n = I18N()
    i18n.load_prompts()
    with pytest.raises(Exception):
        i18n.retrieve("nonexistent_kind", "nonexistent_key")


def test_prompt_file():
    import os

    path = os.path.join(os.path.dirname(__file__), "prompts.json")
    i18n = I18N(prompt_file=path)
    i18n.load_prompts()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)
    assert i18n.retrieve("slices", "role_playing") == "Lorem ipsum dolor sit amet"


def test_prompt_file_env():
    import os

    path = os.path.join(os.path.dirname(__file__), "prompts.json")
    old_env = os.environ.get("CREWAI_PROMPT_FILE")
    try:
        os.environ["CREWAI_PROMPT_FILE"] = path
        i18n = I18N()
        i18n.load_prompts()
        assert i18n.retrieve("slices", "role_playing") == "Lorem ipsum dolor sit amet"
    finally:
        if old_env:
            os.environ["CREWAI_PROMPT_FILE"] = old_env
        else:
            del os.environ["CREWAI_PROMPT_FILE"]

    i18n = I18N()
    i18n.load_prompts()
    assert i18n.retrieve("slices", "role_playing") != "Lorem ipsum dolor sit amet"
