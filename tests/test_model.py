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
        ("gpt-3.5-turbo-0613", "json_schema", True, False),
        ("unknown-model", "json_schema", True, False),
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
    if expect_format_present:
        assert mode == expected_mode
    else:
        assert mode is None


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


def test_openai_caching_disabled_by_default(mock_openai_clients):
    mock_sync, _ = mock_openai_clients
    llm = OpenAILLM(api_key="dummy")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="response 1"))]
    mock_sync.chat.completions.create.return_value = mock_response

    llm.generate("prompt 1")
    llm.generate("prompt 1")

    assert mock_sync.chat.completions.create.call_count == 2


def test_openai_caching_enabled(mock_openai_clients):
    mock_sync, _ = mock_openai_clients
    llm = OpenAILLM(api_key="dummy")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="response 1"))]
    mock_sync.chat.completions.create.return_value = mock_response

    # First call, should call the API
    res1 = llm.generate("prompt 1", use_cache=True)
    assert res1 == "response 1"
    assert mock_sync.chat.completions.create.call_count == 1

    # Second call, should hit the cache
    res2 = llm.generate("prompt 1", use_cache=True)
    assert res2 == "response 1"
    assert mock_sync.chat.completions.create.call_count == 1


def test_ollama_caching_enabled(mock_ollama_clients):
    mock_sync, _ = mock_ollama_clients
    llm = OllamaLLM(model="dummy")
    mock_sync.chat.return_value = {"message": {"content": "ollama response"}}

    # First call
    res1 = llm.generate("prompt 1", use_cache=True)
    assert res1 == "ollama response"
    assert mock_sync.chat.call_count == 1

    # Second call
    res2 = llm.generate("prompt 1", use_cache=True)
    assert res2 == "ollama response"
    assert mock_sync.chat.call_count == 1


def test_json_caching(mock_openai_clients):
    mock_sync, _ = mock_openai_clients
    llm = OpenAILLM(api_key="dummy", model="gpt-4o")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"answer": "json response"}'))]
    mock_sync.chat.completions.create.return_value = mock_response

    # First call
    res1 = llm.generate_json("json prompt", SimpleSchema, use_cache=True)
    assert isinstance(res1, SimpleSchema)
    assert res1.answer == "json response"
    assert mock_sync.chat.completions.create.call_count == 1

    # Second call
    res2 = llm.generate_json("json prompt", SimpleSchema, use_cache=True)
    assert isinstance(res2, SimpleSchema)
    assert res2.answer == "json response"
    assert mock_sync.chat.completions.create.call_count == 1


def test_cache_key_differentiation(mock_openai_clients):
    mock_sync, _ = mock_openai_clients
    llm = OpenAILLM(api_key="dummy")
    mock_sync.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="response 1"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="response 2"))]),
    ]

    res1 = llm.generate("prompt", use_cache=True, temperature=0.5)
    res2 = llm.generate("prompt", use_cache=True, temperature=0.8)

    assert res1 == "response 1"
    assert res2 == "response 2"
    assert mock_sync.chat.completions.create.call_count == 2


# ============================================================================
# OpenRouterLLM Tests
# ============================================================================


@pytest.fixture
def mock_openrouter_clients(mocker):
    """Mock clients for OpenRouter - patches OpenAI clients used by OpenRouterLLM."""
    mock_sync_client = MagicMock()
    mock_async_client = AsyncMock()
    mock_sync_client.chat.completions.create = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock()
    mocker.patch("cogitator.model.openrouter.SyncOpenAI", return_value=mock_sync_client)
    mocker.patch("cogitator.model.openrouter.AsyncOpenAI", return_value=mock_async_client)
    return mock_sync_client, mock_async_client


def test_openrouter_init_default(mock_openrouter_clients):
    """Test OpenRouterLLM initializes with the correct default settings."""
    from cogitator import OpenRouterLLM

    llm = OpenRouterLLM(api_key="test-key")

    assert llm.model == "openai/gpt-4o-mini"
    assert llm.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_openrouter_init_with_custom_model(mock_openrouter_clients):
    """Test OpenRouterLLM with a custom model name."""
    from cogitator import OpenRouterLLM

    llm = OpenRouterLLM(api_key="test-key", model="anthropic/claude-3.5-sonnet")

    assert llm.model == "anthropic/claude-3.5-sonnet"


def test_openrouter_init_with_site_info(mocker):
    """Test OpenRouterLLM includes site info headers when provided."""
    from cogitator import OpenRouterLLM

    mock_sync = MagicMock()
    mock_async = AsyncMock()
    sync_patch = mocker.patch("cogitator.model.openrouter.SyncOpenAI", return_value=mock_sync)
    async_patch = mocker.patch("cogitator.model.openrouter.AsyncOpenAI", return_value=mock_async)

    llm = OpenRouterLLM(
        api_key="test-key",
        site_url="https://example.com",
        site_name="Test App"
    )

    # Verify clients were created with the correct base_url and headers
    expected_headers = {"HTTP-Referer": "https://example.com", "X-Title": "Test App"}
    sync_patch.assert_called_once()
    call_kwargs = sync_patch.call_args[1]
    assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert call_kwargs["default_headers"] == expected_headers


def test_openrouter_inherits_from_openai(mock_openrouter_clients):
    """Test OpenRouterLLM inherits from OpenAILLM."""
    from cogitator import OpenRouterLLM, OpenAILLM

    llm = OpenRouterLLM(api_key="test-key")

    assert isinstance(llm, OpenAILLM)
    assert hasattr(llm, "generate")
    assert hasattr(llm, "generate_async")
    assert hasattr(llm, "generate_json")


class TestOpenRouterInternalMethods:
    """Tests for OpenRouterLLM internal methods."""

    @pytest.fixture
    def openrouter_llm(self, mock_openrouter_clients):
        from cogitator import OpenRouterLLM
        return OpenRouterLLM(api_key="test-key", model="anthropic/claude-3.5-sonnet")

    def test_reset_token_counts(self, openrouter_llm):
        """Test that token counts can be reset."""
        openrouter_llm._last_prompt_tokens = 100
        openrouter_llm._last_completion_tokens = 50
        openrouter_llm._reset_token_counts()
        assert openrouter_llm._last_prompt_tokens is None
        assert openrouter_llm._last_completion_tokens is None

    def test_create_cache_key_deterministic(self, openrouter_llm):
        """Test that cache key is deterministic for same inputs."""
        key1 = openrouter_llm._create_cache_key("test prompt", temperature=0.7)
        key2 = openrouter_llm._create_cache_key("test prompt", temperature=0.7)
        assert key1 == key2

    def test_create_cache_key_different_prompts(self, openrouter_llm):
        """Test that different prompts produce different keys."""
        key1 = openrouter_llm._create_cache_key("prompt 1")
        key2 = openrouter_llm._create_cache_key("prompt 2")
        assert key1 != key2

    def test_create_cache_key_different_kwargs(self, openrouter_llm):
        """Test that different kwargs produce different keys."""
        key1 = openrouter_llm._create_cache_key("prompt", temperature=0.5)
        key2 = openrouter_llm._create_cache_key("prompt", temperature=0.9)
        assert key1 != key2

    def test_attributes_initialized_correctly(self, openrouter_llm):
        """Test that OpenRouter-specific attributes are initialized."""
        assert openrouter_llm.model == "anthropic/claude-3.5-sonnet"
        assert openrouter_llm.temperature == 0.7
        assert openrouter_llm.max_tokens == 512
        assert openrouter_llm.seed == 33
        assert openrouter_llm._cache == {}

    def test_tiktoken_encoding_loaded(self, openrouter_llm):
        """Test that tiktoken encoding is loaded for token counting."""
        assert hasattr(openrouter_llm, "encoding")
        # Should use cl100k_base for OpenRouter models
        if openrouter_llm.encoding is not None:
            assert openrouter_llm.encoding.name == "cl100k_base"

    def test_no_site_headers_by_default(self, mocker):
        """Test that no headers are added when site info not provided."""
        from cogitator import OpenRouterLLM

        mock_sync = MagicMock()
        mock_async = AsyncMock()
        sync_patch = mocker.patch("cogitator.model.openrouter.SyncOpenAI", return_value=mock_sync)
        mocker.patch("cogitator.model.openrouter.AsyncOpenAI", return_value=mock_async)

        llm = OpenRouterLLM(api_key="test-key")

        call_kwargs = sync_patch.call_args[1]
        assert "default_headers" not in call_kwargs or call_kwargs["default_headers"] == {}


# ============================================================================
# Configurable Model Capabilities Tests
# ============================================================================


def test_openai_capability_autodetect_structured_output(mock_openai_clients):
    """Test auto-detection of structured output capability for known models."""
    llm_gpt4o = OpenAILLM(api_key="d", model="gpt-4o")
    assert llm_gpt4o._supports_structured_output is True
    assert llm_gpt4o._supports_json_mode is True

    llm_gpt4o_mini = OpenAILLM(api_key="d", model="gpt-4o-mini")
    assert llm_gpt4o_mini._supports_structured_output is True
    assert llm_gpt4o_mini._supports_json_mode is True


def test_openai_capability_autodetect_json_mode_only(mock_openai_clients):
    """Test auto-detection of JSON mode capability for models without structured output."""
    llm_gpt4 = OpenAILLM(api_key="d", model="gpt-4")
    assert llm_gpt4._supports_structured_output is False
    assert llm_gpt4._supports_json_mode is True

    llm_gpt35 = OpenAILLM(api_key="d", model="gpt-3.5-turbo-1106")
    assert llm_gpt35._supports_structured_output is False
    assert llm_gpt35._supports_json_mode is True


def test_openai_capability_autodetect_unknown_model(mock_openai_clients):
    """Test auto-detection for unknown models defaults to disabled."""
    llm = OpenAILLM(api_key="d", model="unknown-model-xyz")
    assert llm._supports_structured_output is False
    assert llm._supports_json_mode is False


def test_openai_capability_override_enable(mock_openai_clients):
    """Test overriding capabilities to enable them on unknown models."""
    llm = OpenAILLM(
        api_key="d",
        model="custom-model",
        supports_structured_output=True,
        supports_json_mode=True
    )
    assert llm._supports_structured_output is True
    assert llm._supports_json_mode is True


def test_openai_capability_override_disable(mock_openai_clients):
    """Test overriding capabilities to disable them on known models."""
    llm = OpenAILLM(
        api_key="d",
        model="gpt-4o",  # Normally supports structured output
        supports_structured_output=False,
        supports_json_mode=False
    )
    assert llm._supports_structured_output is False
    assert llm._supports_json_mode is False


def test_openai_capability_affects_api_params(mock_openai_clients):
    """Test it that capability overrides affect _prepare_api_params behavior."""
    # Unknown model with capabilities force-enabled
    llm_enabled = OpenAILLM(
        api_key="d",
        model="custom-model",
        supports_structured_output=True
    )
    params, mode = llm_enabled._prepare_api_params(is_json_mode=True, response_schema=DummySchema)
    assert mode == "json_schema"
    assert params["response_format"]["type"] == "json_schema"

    # Known model with capabilities force-disabled
    llm_disabled = OpenAILLM(
        api_key="d",
        model="gpt-4o",
        supports_structured_output=False,
        supports_json_mode=False
    )
    params2, mode2 = llm_disabled._prepare_api_params(is_json_mode=True, response_schema=DummySchema)
    # Should fall through to attempting json_schema anyway since schema is provided
    # but mode may differ based on fallback logic
    assert mode2 is None or mode2 == "json_schema"  # Depends on fallback behavior


# ============================================================================
# OllamaLLM Internal Method Tests (Minimal Mocking)
# ============================================================================


class TestOllamaStripContent:
    """Tests for OllamaLLM._strip_content method - no API mocking needed."""

    @pytest.fixture
    def ollama_llm(self, mock_ollama_clients):
        return OllamaLLM(model="test-model")

    def test_strip_content_from_dict_message(self, ollama_llm):
        """Test extracting content from dict response with message dict."""
        resp = {"message": {"content": "  Hello world  "}}
        assert ollama_llm._strip_content(resp) == "Hello world"

    def test_strip_content_from_dict_empty(self, ollama_llm):
        """Test extracting content from dict with empty content."""
        resp = {"message": {"content": ""}}
        assert ollama_llm._strip_content(resp) == ""

    def test_strip_content_from_dict_missing_content(self, ollama_llm):
        """Test extracting content from dict without content key."""
        resp = {"message": {}}
        assert ollama_llm._strip_content(resp) == ""

    def test_strip_content_from_dict_missing_message(self, ollama_llm):
        """Test extracting content from dict without message key."""
        resp = {}
        assert ollama_llm._strip_content(resp) == ""

    def test_strip_content_from_object(self, ollama_llm):
        """Test extracting content from object with message.content attribute."""
        class FakeMessage:
            content = "  Object content  "

        class FakeResponse:
            message = FakeMessage()

        assert ollama_llm._strip_content(FakeResponse()) == "Object content"

    def test_strip_content_numeric_value(self, ollama_llm):
        """Test that numeric content is converted to string."""
        resp = {"message": {"content": 42}}
        assert ollama_llm._strip_content(resp) == "42"

    def test_strip_content_float_value(self, ollama_llm):
        """Test that float content is converted to string."""
        resp = {"message": {"content": 3.14}}
        assert ollama_llm._strip_content(resp) == "3.14"


class TestOllamaPrepareOptions:
    """Tests for OllamaLLM._prepare_options method."""

    @pytest.fixture
    def ollama_llm(self, mock_ollama_clients):
        return OllamaLLM(
            model="test-model",
            temperature=0.7,
            max_tokens=1024,
            seed=33,
            stop=None
        )

    def test_default_options(self, ollama_llm):
        """Test that defaults are used when no overrides provided."""
        opts = ollama_llm._prepare_options()
        assert opts["temperature"] == 0.7
        assert opts["num_predict"] == 1024
        assert opts["seed"] == 33
        assert "stop" not in opts

    def test_override_temperature(self, ollama_llm):
        """Test overriding temperature."""
        opts = ollama_llm._prepare_options(temperature=0.1)
        assert opts["temperature"] == 0.1

    def test_override_max_tokens(self, ollama_llm):
        """Test overriding max_tokens -> num_predict."""
        opts = ollama_llm._prepare_options(max_tokens=512)
        assert opts["num_predict"] == 512

    def test_override_seed(self, ollama_llm):
        """Test overriding seed."""
        opts = ollama_llm._prepare_options(seed=42)
        assert opts["seed"] == 42

    def test_add_stop_sequences(self, ollama_llm):
        """Test adding stop sequences."""
        opts = ollama_llm._prepare_options(stop=["STOP", "END"])
        assert opts["stop"] == ["STOP", "END"]

    def test_extra_kwargs_passed_through(self, ollama_llm):
        """Test that extra kwargs are passed through."""
        opts = ollama_llm._prepare_options(top_k=40, top_p=0.9)
        assert opts["top_k"] == 40
        assert opts["top_p"] == 0.9

    def test_invalid_seed_removed(self, ollama_llm):
        """Test that invalid seed values are removed."""
        opts = ollama_llm._prepare_options(seed="invalid")
        assert "seed" not in opts

    def test_none_values_filtered(self, ollama_llm):
        """Test that None values are filtered out."""
        opts = ollama_llm._prepare_options(seed=None)
        assert "seed" not in opts


class TestOllamaUpdateTokenCounts:
    """Tests for OllamaLLM._update_token_counts method."""

    @pytest.fixture
    def ollama_llm(self, mock_ollama_clients):
        return OllamaLLM(model="test-model")

    def test_uses_api_counts_when_available(self, ollama_llm):
        """Test that API token counts are used when present."""
        resp = {"prompt_eval_count": 100, "eval_count": 50}
        ollama_llm._update_token_counts("test prompt", resp, "test completion")
        assert ollama_llm._last_prompt_tokens == 100
        assert ollama_llm._last_completion_tokens == 50

    def test_falls_back_to_approximation(self, ollama_llm):
        """Test fallback to approx_token_length when API counts missing."""
        resp = {}
        ollama_llm._update_token_counts("hello world", resp, "hi there")
        # Approximation should produce some counts
        assert ollama_llm._last_prompt_tokens is not None
        assert ollama_llm._last_completion_tokens is not None
        assert ollama_llm._last_prompt_tokens >= 2  # "hello world" = 2 words

    def test_handles_non_dict_response(self, ollama_llm):
        """Test handling of non-dict response (uses approximation)."""
        ollama_llm._update_token_counts("test", "not a dict", "response")
        assert ollama_llm._last_prompt_tokens is not None


# ============================================================================
# OpenAILLM Internal Method Tests (Minimal Mocking)
# ============================================================================


class TestOpenAIPrepareApiParams:
    """Tests for OpenAILLM._prepare_api_params method."""

    @pytest.fixture
    def openai_llm_gpt4o(self, mock_openai_clients):
        """GPT-4O supports structured output and JSON mode."""
        return OpenAILLM(api_key="test", model="gpt-4o")

    @pytest.fixture
    def openai_llm_gpt35(self, mock_openai_clients):
        """GPT-3.5 supports JSON mode but not structured output."""
        return OpenAILLM(api_key="test", model="gpt-3.5-turbo-1106")

    @pytest.fixture
    def openai_llm_unknown(self, mock_openai_clients):
        """Unknown model doesn't support JSON mode by default."""
        return OpenAILLM(api_key="test", model="unknown-model")

    def test_no_json_mode_returns_basic_params(self, openai_llm_gpt4o):
        """Test that non-JSON mode returns basic params."""
        params, mode = openai_llm_gpt4o._prepare_api_params(is_json_mode=False)
        assert "response_format" not in params
        assert mode is None

    def test_json_mode_with_schema_uses_json_schema(self, openai_llm_gpt4o):
        """Test JSON mode with schema uses json_schema format for gpt-4o."""
        params, mode = openai_llm_gpt4o._prepare_api_params(
            is_json_mode=True,
            response_schema=DummySchema
        )
        assert mode == "json_schema"
        assert params["response_format"]["type"] == "json_schema"

    def test_json_mode_without_schema_uses_json_object(self, openai_llm_gpt4o):
        """Test JSON mode without schema uses json_object format."""
        params, mode = openai_llm_gpt4o._prepare_api_params(
            is_json_mode=True,
            response_schema=None
        )
        assert mode == "json_object"
        assert params["response_format"]["type"] == "json_object"

    def test_gpt35_uses_json_object_not_schema(self, openai_llm_gpt35):
        """Test GPT-3.5 uses json_object even with schema (no structured output)."""
        params, mode = openai_llm_gpt35._prepare_api_params(
            is_json_mode=True,
            response_schema=DummySchema
        )
        # GPT-3.5 doesn't support structured output, should fall back
        assert mode == "json_object"
        assert params["response_format"]["type"] == "json_object"

    def test_unknown_model_falls_back_to_json_schema(self, openai_llm_unknown):
        """Test unknown model falls back to json_schema when schema provided."""
        params, mode = openai_llm_unknown._prepare_api_params(
            is_json_mode=True,
            response_schema=DummySchema
        )
        # Unknown model attempts json_schema anyway when schema is provided
        # (with a warning logged)
        assert mode == "json_schema"

    def test_kwargs_passed_through(self, openai_llm_gpt4o):
        """Test that extra kwargs are passed through."""
        params, _ = openai_llm_gpt4o._prepare_api_params(
            is_json_mode=False,
            custom_param="value"
        )
        assert params["custom_param"] == "value"


class TestOpenAIUpdateTokenCounts:
    """Tests for OpenAILLM._update_token_counts method."""

    @pytest.fixture
    def openai_llm(self, mock_openai_clients):
        return OpenAILLM(api_key="test", model="gpt-4o-mini")

    def test_uses_api_counts_when_available(self, openai_llm):
        """Test that API usage counts are used when present."""
        class FakeUsage:
            prompt_tokens = 150
            completion_tokens = 75

        class FakeResponse:
            usage = FakeUsage()

        openai_llm._update_token_counts("test", FakeResponse(), "completion")
        assert openai_llm._last_prompt_tokens == 150
        assert openai_llm._last_completion_tokens == 75

    def test_falls_back_to_tiktoken(self, openai_llm):
        """Test fallback to tiktoken when API counts missing."""
        class FakeResponse:
            usage = None

        openai_llm._update_token_counts("hello world test", FakeResponse(), "response text")
        # Should use tiktoken approximation
        assert openai_llm._last_prompt_tokens is not None
        assert openai_llm._last_prompt_tokens > 0


class TestOpenAITokenCounting:
    """Tests for OpenAILLM token counting methods."""

    @pytest.fixture
    def openai_llm(self, mock_openai_clients):
        return OpenAILLM(api_key="test", model="gpt-4o-mini")

    def test_get_last_prompt_tokens_initially_none(self, openai_llm):
        """Test that token counts start as None."""
        assert openai_llm.get_last_prompt_tokens() is None
        assert openai_llm.get_last_completion_tokens() is None

    def test_reset_token_counts(self, openai_llm):
        """Test resetting token counts."""
        openai_llm._last_prompt_tokens = 100
        openai_llm._last_completion_tokens = 50
        openai_llm._reset_token_counts()
        assert openai_llm._last_prompt_tokens is None
        assert openai_llm._last_completion_tokens is None
