from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from cogitator.model import BaseLLM, OpenAILLM, OllamaLLM


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
    mocker.patch("openai.OpenAI", return_value=mock_sync_client)
    mocker.patch("openai.AsyncOpenAI", return_value=mock_async_client)
    return mock_sync_client, mock_async_client


@pytest.fixture
def mock_ollama_clients(mocker):
    mock_sync_client = MagicMock()
    mock_async_client = AsyncMock()
    mock_sync_client.chat = MagicMock()
    mock_async_client.chat = AsyncMock()
    mock_sync_client.list = MagicMock(return_value={'models': []})
    mocker.patch("cogitator.model.Client", new_callable=MagicMock, return_value=mock_sync_client)
    mocker.patch("cogitator.model.AsyncClient", new_callable=AsyncMock,
                 return_value=mock_async_client)
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


def test_extract_json_block():
    with patch("cogitator.model.Client") as MockClient:
        MockClient.return_value.list.return_value = {'models': []}
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
    "model_name, expected_mode, expect_format_present, expect_additional_props_false", [
        ("gpt-4o", "json_schema", True, True),
        ("gpt-4o-mini", "json_schema", True, True),
        ("gpt-4-turbo", "json_object", True, False),
        ("gpt-3.5-turbo-1106", "json_object", True, False),
        ("gpt-3.5-turbo-0613", "json_schema", True, False),
        ("unknown-model", "json_schema", True, False),
    ])
def test_openai_prepare_api_params_json_modes(mock_openai_clients, model_name, expected_mode,
                                              expect_format_present, expect_additional_props_false):
    llm = OpenAILLM(api_key="dummy_key", model=model_name)
    params, mode = llm._prepare_api_params(is_json_mode=True, response_schema=DummySchema)

    if expect_format_present:
        assert "response_format" in params
        if expected_mode == "json_schema":
            assert params["response_format"]["type"] == "json_schema"
            assert "json_schema" in params["response_format"]
            assert params["response_format"]["json_schema"]["name"] == "DummySchema"
            if expect_additional_props_false:
                assert params["response_format"]["json_schema"]["schema"].get(
                    "additionalProperties") is False
            else:
                assert params["response_format"]["json_schema"]["schema"].get(
                    "additionalProperties") is not False
        elif expected_mode == "json_object":
            assert params["response_format"]["type"] == "json_object"
            assert "json_schema" not in params["response_format"]
    else:
        assert "response_format" not in params

    assert mode == expected_mode if expect_format_present else mode is None


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


def test_ollama_init_success(mocker):
    mock_sync = MagicMock()
    mock_async = AsyncMock()
    mock_sync.list = MagicMock(return_value={'models': []})

    # Using local patch with return_value as it seemed most reliable for sync
    with patch("cogitator.model.Client", return_value=mock_sync) as patched_client, \
        patch("cogitator.model.AsyncClient", return_value=mock_async) as patched_async_client:
        llm = OllamaLLM(model="ollama-test", ollama_host="http://testhost:11434")
        assert llm.model == "ollama-test"
        assert llm.host == "http://testhost:11434"

        # Check sync client
        assert isinstance(llm._client, MagicMock)
        assert llm._client == mock_sync

        # --- FIX: Avoid isinstance check for AsyncMock, just check identity ---
        # This assumes the patch mechanism *should* assign the mock object directly.
        # If this fails, it points to a very subtle patching/mocking issue.
        assert llm._async_client == mock_async
        # --- End FIX ---

        # Verify constructors were called
        patched_client.assert_called_once_with(host="http://testhost:11434")
        patched_async_client.assert_called_once_with(host="http://testhost:11434")


def test_ollama_strip_content():
    with patch("cogitator.model.Client") as MockClient:
        MockClient.return_value.list.return_value = {'models': []}
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


def test_ollama_prepare_options():
    with patch("cogitator.model.Client") as MockClient:
        MockClient.return_value.list.return_value = {'models': []}
        llm = OllamaLLM(model="d", temperature=0.5, max_tokens=100, stop=["\n"], seed=1)

    opts = llm._prepare_options(temperature=0.8, seed=None, stop=["stop"], extra_param=True)
    assert opts == {"temperature": 0.8, "num_predict": 100, "stop": ["stop"], "extra_param": True}

    opts_defaults = llm._prepare_options()
    assert opts_defaults == {"temperature": 0.5, "num_predict": 100, "seed": 1, "stop": ["\n"]}
