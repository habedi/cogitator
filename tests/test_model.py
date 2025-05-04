from unittest.mock import MagicMock, AsyncMock

import pytest
from pydantic import BaseModel, Field

from cogitator import BaseLLM
from cogitator import OllamaLLM
from cogitator import OpenAILLM


class DummySchema(BaseModel):
    name: str = Field(...)
    value: int = Field(..., gt=0)


class SimpleSchema(BaseModel):
    answer: str


@pytest.fixture
def mock_openai_clients(mocker):
    mock_sync_client = MagicMock()
    mock_async_client = AsyncMock()
    mock_sync_client.chat.completions.create = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock()
    mocker.patch("cogitator.model.openai.SyncOpenAI", return_value=mock_sync_client)
    mocker.patch("cogitator.model.openai.AsyncOpenAI", return_value=mock_async_client)
    return mock_sync_client, mock_async_client


@pytest.fixture
def mock_ollama_clients(mocker):
    mock_sync_client = MagicMock()
    mock_async_client = AsyncMock()
    mock_sync_client.chat = MagicMock()
    mock_async_client.chat = AsyncMock()
    mock_sync_client.list = MagicMock(return_value={'models': []})
    mocker.patch("cogitator.model.ollama.Client", new_callable=MagicMock, return_value=mock_sync_client)
    mocker.patch("cogitator.model.ollama.AsyncClient", return_value=mock_async_client)
    return mock_sync_client, mock_async_client


def test_base_llm_abstract_methods():
    class ConcreteLLM(BaseLLM):
        def generate(self, prompt: str, **kwargs) -> str: return ""

        async def generate_async(self, prompt: str, **kwargs) -> str: return ""

        def generate_stream(self, prompt: str, **kwargs): yield ""

        async def generate_stream_async(self, prompt: str, **kwargs): yield ""

        def _generate_json_internal(self, prompt: str, response_model, **kwargs): return "{}", None

        async def _generate_json_internal_async(self, prompt: str, response_model,
                                                **kwargs): return "{}", None

    instance = ConcreteLLM()
    assert hasattr(instance, "generate")


def test_extract_json_block(mock_ollama_clients):
    llm = OllamaLLM(model="dummy")

    text_fence = "```json\n{\"key\": \"value\"}\n```"
    text_fence_no_lang = "```\n{\"key\": \"value2\"}\n```"
    text_braces = "Some text {\"key\": \"value3\"} more text"
    text_brackets = "Some text [{\"key\": \"value4\"}] more text"
    text_both = "Text {\"a\":1} then [\"b\"] end"
    text_nested = "Text {\"a\": {\"b\": 1}} end"
    text_no_json = "Just plain text"
    text_empty = ""

    assert llm._extract_json_block(text_fence) == "{\"key\": \"value\"}"
    assert llm._extract_json_block(text_fence_no_lang) == "{\"key\": \"value2\"}"
    assert llm._extract_json_block(text_braces) == "{\"key\": \"value3\"}"
    assert llm._extract_json_block(text_brackets) == "[{\"key\": \"value4\"}]"
    assert llm._extract_json_block(text_both) == "{\"a\":1}"
    assert llm._extract_json_block(text_nested) == "{\"a\": {\"b\": 1}}"
    assert llm._extract_json_block(text_no_json) == "Just plain text"
    assert llm._extract_json_block(text_empty) == ""


@pytest.mark.parametrize(
    "model_name, expected_mode, expect_format_present, expect_additional_props", [
        ("gpt-4o", "json_schema", True, False),
        ("gpt-4o-mini", "json_schema", True, False),
        ("gpt-4-turbo", "json_object", True, None),
        ("gpt-3.5-turbo-1106", "json_object", True, None),
        ("gpt-3.5-turbo-0613", "json_schema", True, None),
        ("unknown-model", "json_schema", True, None),
    ])
def test_openai_prepare_api_params_json_modes(mock_openai_clients, model_name, expected_mode,
                                              expect_format_present, expect_additional_props):
    llm = OpenAILLM(api_key="dummy_key", model=model_name)
    params, mode = llm._prepare_api_params(is_json_mode=True, response_schema=DummySchema)

    if expect_format_present:
        assert "response_format" in params
        rf = params["response_format"]
        assert rf["type"] == expected_mode
        if expected_mode == "json_schema":
            assert "json_schema" in rf
            assert rf["json_schema"]["name"] == "DummySchema"
            schema = rf["json_schema"]["schema"]
            assert schema.get("additionalProperties") is expect_additional_props
        elif expected_mode == "json_object":
            assert "json_schema" not in rf
            assert expect_additional_props is None
    else:
        assert "response_format" not in params
        assert expect_additional_props is None

    assert mode == expected_mode


def test_openai_prepare_api_params_no_schema_json_mode(mock_openai_clients):
    llm_json = OpenAILLM(api_key="d", model="gpt-4-turbo")
    params, mode = llm_json._prepare_api_params(is_json_mode=True, response_schema=None)
    assert params["response_format"]["type"] == "json_object"
    assert mode == "json_object"

    llm_no_json = OpenAILLM(api_key="d", model="gpt-3.5-turbo-0613")
    params, mode = llm_no_json._prepare_api_params(is_json_mode=True, response_schema=None)
    assert "response_format" not in params
    assert mode is None


def test_openai_prepare_api_params_schema_generation_fails(mock_openai_clients, mocker):
    llm = OpenAILLM(api_key="d", model="gpt-4o")
    mocker.patch.object(DummySchema, "model_json_schema", side_effect=TypeError("Schema fail"))
    params, mode = llm._prepare_api_params(is_json_mode=True, response_schema=DummySchema)
    assert params["response_format"]["type"] == "json_object"
    assert mode == "json_object"

    llm_no_fallback = OpenAILLM(api_key="d", model="gpt-3.5-turbo-0613")
    mocker.patch.object(DummySchema, "model_json_schema",
                        side_effect=TypeError("Schema fail again"))
    params_no_fallback, mode_no_fallback = llm_no_fallback._prepare_api_params(is_json_mode=True,
                                                                               response_schema=DummySchema)
    assert "response_format" not in params_no_fallback
    assert mode_no_fallback is None


def test_ollama_init_success(mock_ollama_clients):
    mock_sync, mock_async = mock_ollama_clients
    llm = OllamaLLM(model="ollama-test", ollama_host="http://testhost:11434")
    assert llm.model == "ollama-test"
    assert llm.host == "http://testhost:11434"
    assert llm._client == mock_sync
    assert llm._async_client == mock_async

    # Verify constructors were called by the patcher via the fixture
    # Check the call args on the mock returned by the patcher
    from cogitator.model.ollama import Client, AsyncClient
    Client.assert_called_once_with(host="http://testhost:11434")
    AsyncClient.assert_called_once_with(host="http://testhost:11434")


def test_ollama_strip_content(mock_ollama_clients):
    llm = OllamaLLM(model="dummy")

    response_dict = {"message": {"content": "  Strip Me!  "}}
    response_obj = MagicMock(message=MagicMock(content="  Strip Me Too!  "))
    response_bad_obj = MagicMock(message=None)
    response_bad_dict = {"message": None}
    response_no_content = {"message": {"role": "assistant"}}

    assert llm._strip_content(response_dict) == "Strip Me!"
    assert llm._strip_content(response_obj) == "Strip Me Too!"
    assert llm._strip_content(response_bad_obj) == ""
    assert llm._strip_content(response_bad_dict) == ""
    assert llm._strip_content(response_no_content) == ""
    assert llm._strip_content(None) == ""
    assert llm._strip_content("string") == ""


def test_ollama_prepare_options(mock_ollama_clients):
    llm = OllamaLLM(model="d", temperature=0.5, max_tokens=100, stop=["\n"], seed=1)

    opts = llm._prepare_options(temperature=0.8, seed=None, stop=["stop"], extra_param=True)
    assert opts == {"temperature": 0.8, "num_predict": 100, "stop": ["stop"], "extra_param": True}

    opts_defaults = llm._prepare_options()
    assert opts_defaults == {"temperature": 0.5, "num_predict": 100, "seed": 1, "stop": ["\n"]}
