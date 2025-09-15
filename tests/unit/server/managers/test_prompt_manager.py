import asyncio
from unittest.mock import Mock

import mcp.types as types
import pytest
from mcp.server.lowlevel.server import Server

from minimcp.server.managers.prompt_manager import PromptDefinition, PromptManager


class TestPromptManager:
    """Test suite for PromptManager class."""

    @pytest.fixture
    def mock_core(self) -> Mock:
        """Create a mock Server for testing."""
        core = Mock(spec=Server)
        core.list_prompts = Mock(return_value=Mock())
        core.get_prompt = Mock(return_value=Mock())
        return core

    @pytest.fixture
    def prompt_manager(self, mock_core: Mock) -> PromptManager:
        """Create a PromptManager instance with mocked core."""
        return PromptManager(mock_core)

    def test_init_hooks_core_methods(self, mock_core: Mock):
        """Test that PromptManager properly hooks into Server methods."""
        prompt_manager = PromptManager(mock_core)

        # Verify that core methods were called to register handlers
        mock_core.list_prompts.assert_called_once()
        mock_core.get_prompt.assert_called_once()

        # Verify internal state
        assert prompt_manager._prompts == {}

    def test_add_prompt_basic_function(self, prompt_manager: PromptManager):
        """Test adding a basic function as a prompt."""

        def sample_prompt(topic: str) -> str:
            """A sample prompt for testing."""
            return f"Tell me about {topic}"

        result = prompt_manager.add(sample_prompt)

        # Verify the returned prompt
        assert isinstance(result, types.Prompt)
        assert result.name == "sample_prompt"
        assert result.description == "A sample prompt for testing."
        assert result.arguments is not None
        assert len(result.arguments) == 1
        assert result.arguments[0].name == "topic"
        assert result.arguments[0].required is True
        assert result.title is None
        assert result.meta is None

        # Verify internal state
        assert "sample_prompt" in prompt_manager._prompts
        prompt, func = prompt_manager._prompts["sample_prompt"]
        assert prompt == result

    def test_add_prompt_with_custom_options(self, prompt_manager: PromptManager):
        """Test adding a prompt with custom name, title, description, and metadata."""

        def basic_func(query: str) -> str:
            return f"Search for: {query}"

        custom_meta = {"version": "1.0", "category": "search"}

        result = prompt_manager.add(
            basic_func,
            name="custom_search",
            title="Custom Search Prompt",
            description="Custom description for search",
            meta=custom_meta,
        )

        assert result.name == "custom_search"
        assert result.title == "Custom Search Prompt"
        assert result.description == "Custom description for search"
        assert result.meta == custom_meta

        # Verify it's stored with custom name
        assert "custom_search" in prompt_manager._prompts
        assert "basic_func" not in prompt_manager._prompts

    def test_add_prompt_without_docstring(self, prompt_manager: PromptManager):
        """Test adding a prompt without docstring uses None as description."""

        def no_doc_prompt(text: str) -> str:
            return text

        result = prompt_manager.add(no_doc_prompt)
        assert result.description is None

    def test_add_async_prompt(self, prompt_manager: PromptManager):
        """Test adding an async function as a prompt."""

        async def async_prompt(delay: float, message: str) -> str:
            """An async prompt."""
            await asyncio.sleep(delay)
            return f"Delayed message: {message}"

        result = prompt_manager.add(async_prompt)

        assert result.name == "async_prompt"
        assert result.description == "An async prompt."
        assert "async_prompt" in prompt_manager._prompts

    def test_add_prompt_with_optional_parameters(self, prompt_manager: PromptManager):
        """Test adding a prompt with optional parameters."""

        def prompt_with_defaults(required_param: str, optional_param: str = "default") -> str:
            """A prompt with optional parameters."""
            return f"{required_param} - {optional_param}"

        result = prompt_manager.add(prompt_with_defaults)

        assert result.arguments is not None
        assert len(result.arguments) == 2

        # Find arguments by name
        required_arg = next(arg for arg in result.arguments if arg.name == "required_param")
        optional_arg = next(arg for arg in result.arguments if arg.name == "optional_param")

        assert required_arg.required is True
        assert optional_arg.required is False

    def test_add_duplicate_prompt_raises_error(self, prompt_manager: PromptManager):
        """Test that adding a prompt with duplicate name raises ValueError."""

        def prompt1(x: str) -> str:
            return x

        def prompt2(y: str) -> str:
            return y

        # Add first prompt
        prompt_manager.add(prompt1, name="duplicate_name")

        # Adding second prompt with same name should raise error
        with pytest.raises(ValueError, match="Prompt duplicate_name already registered"):
            prompt_manager.add(prompt2, name="duplicate_name")

    def test_add_again_prompt_raises_error(self, prompt_manager: PromptManager):
        """Test that adding a prompt again raises ValueError."""

        def prompt1(x: str) -> str:
            return x

        # Add prompt
        prompt_manager.add(prompt1)

        # Adding prompt again should raise error
        with pytest.raises(ValueError, match="Prompt prompt1 already registered"):
            prompt_manager.add(prompt1)

    def test_remove_existing_prompt(self, prompt_manager: PromptManager):
        """Test removing an existing prompt."""

        def test_prompt(x: str) -> str:
            return x

        # Add prompt first
        added_prompt = prompt_manager.add(test_prompt)
        assert "test_prompt" in prompt_manager._prompts

        # Remove the prompt
        removed_prompt = prompt_manager.remove("test_prompt")

        assert removed_prompt == added_prompt
        assert "test_prompt" not in prompt_manager._prompts

    def test_remove_nonexistent_prompt_raises_error(self, prompt_manager: PromptManager):
        """Test that removing a non-existent prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt nonexistent not found"):
            prompt_manager.remove("nonexistent")

    @pytest.mark.asyncio
    async def test_list_prompts_empty(self, prompt_manager: PromptManager):
        """Test listing prompts when no prompts are registered."""
        result = prompt_manager.list()
        assert result == []

        # Test async version
        async_result = await prompt_manager._async_list()
        assert async_result == []

    @pytest.mark.asyncio
    async def test_list_prompts_with_multiple_prompts(self, prompt_manager: PromptManager):
        """Test listing prompts when multiple prompts are registered."""

        def prompt1(x: str) -> str:
            return x

        def prompt2(y: str) -> str:
            return y

        added_prompt1 = prompt_manager.add(prompt1)
        added_prompt2 = prompt_manager.add(prompt2)

        result = prompt_manager.list()

        assert len(result) == 2
        assert added_prompt1 in result
        assert added_prompt2 in result

        # Test async version
        async_result = await prompt_manager._async_list()
        assert len(async_result) == 2
        assert added_prompt1 in async_result
        assert added_prompt2 in async_result

    @pytest.mark.asyncio
    async def test_get_prompt_sync_function(self, prompt_manager: PromptManager):
        """Test getting a synchronous prompt."""

        def greeting_prompt(name: str, greeting: str = "Hello") -> str:
            """Generate a greeting message."""
            return f"{greeting}, {name}!"

        prompt_manager.add(greeting_prompt)

        result = await prompt_manager.get("greeting_prompt", {"name": "Alice"})

        assert isinstance(result, types.GetPromptResult)
        assert result.description == "Generate a greeting message."
        assert len(result.messages) == 1
        assert isinstance(result.messages[0], types.PromptMessage)
        assert result.messages[0].role == "user"
        assert isinstance(result.messages[0].content, types.TextContent)
        assert result.messages[0].content.text == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_get_prompt_async_function(self, prompt_manager: PromptManager):
        """Test getting an asynchronous prompt."""

        async def async_greeting_prompt(name: str) -> str:
            """Async generate a greeting message."""
            await asyncio.sleep(0.01)  # Small delay to make it actually async
            return f"Hello, {name}!"

        prompt_manager.add(async_greeting_prompt)

        result = await prompt_manager.get("async_greeting_prompt", {"name": "Bob"})

        assert isinstance(result, types.GetPromptResult)
        assert result.description == "Async generate a greeting message."
        assert len(result.messages) == 1
        assert isinstance(result.messages[0].content, types.TextContent)
        assert result.messages[0].content.text == "Hello, Bob!"

    @pytest.mark.asyncio
    async def test_get_prompt_with_all_arguments(self, prompt_manager: PromptManager):
        """Test getting a prompt with all arguments provided."""

        def detailed_prompt(topic: str, style: str = "formal", length: str = "short") -> str:
            """Generate a detailed prompt."""
            return f"Write a {length} {style} piece about {topic}"

        prompt_manager.add(detailed_prompt)

        result = await prompt_manager.get("detailed_prompt", {"topic": "AI", "style": "casual", "length": "long"})

        assert isinstance(result.messages[0].content, types.TextContent)
        assert result.messages[0].content.text == "Write a long casual piece about AI"

    @pytest.mark.asyncio
    async def test_get_nonexistent_prompt_raises_error(self, prompt_manager: PromptManager):
        """Test that getting a non-existent prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt nonexistent not found"):
            await prompt_manager.get("nonexistent", {})

    @pytest.mark.asyncio
    async def test_get_prompt_missing_required_arguments(self, prompt_manager: PromptManager):
        """Test that getting a prompt with missing required arguments raises ValueError."""

        def strict_prompt(required_param: str, optional_param: str = "default") -> str:
            """A prompt with required parameters."""
            return f"{required_param} - {optional_param}"

        prompt_manager.add(strict_prompt)

        with pytest.raises(ValueError, match="Missing required arguments"):
            await prompt_manager.get("strict_prompt", {})

        with pytest.raises(ValueError, match="Missing required arguments"):
            await prompt_manager.get("strict_prompt", {"optional_param": "value"})

    @pytest.mark.asyncio
    async def test_get_prompt_returns_prompt_message_list(self, prompt_manager: PromptManager):
        """Test that prompt function returning PromptMessage list works correctly."""

        def multi_message_prompt(topic: str) -> list[types.PromptMessage]:
            """Generate multiple messages."""
            return [
                types.PromptMessage(
                    role="assistant", content=types.TextContent(type="text", text="I'm ready to help you.")
                ),
                types.PromptMessage(role="user", content=types.TextContent(type="text", text=f"Tell me about {topic}")),
            ]

        prompt_manager.add(multi_message_prompt)

        result = await prompt_manager.get("multi_message_prompt", {"topic": "Python"})

        assert len(result.messages) == 2
        assert result.messages[0].role == "assistant"
        assert isinstance(result.messages[0].content, types.TextContent)
        assert result.messages[0].content.text == "I'm ready to help you."
        assert result.messages[1].role == "user"
        assert isinstance(result.messages[1].content, types.TextContent)
        assert result.messages[1].content.text == "Tell me about Python"

    @pytest.mark.asyncio
    async def test_get_prompt_returns_dict_messages(self, prompt_manager: PromptManager):
        """Test that prompt function returning dict messages works correctly."""

        def dict_message_prompt(question: str) -> list[dict]:
            """Generate messages as dicts."""
            return [{"role": "user", "content": {"type": "text", "text": question}}]

        prompt_manager.add(dict_message_prompt)

        result = await prompt_manager.get("dict_message_prompt", {"question": "What is AI?"})

        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert isinstance(result.messages[0].content, types.TextContent)
        assert result.messages[0].content.text == "What is AI?"

    @pytest.mark.asyncio
    async def test_get_prompt_returns_complex_data(self, prompt_manager: PromptManager):
        """Test that prompt function returning complex data structures works correctly."""

        def complex_data_prompt(data_type: str) -> str:
            """Generate complex data as string."""
            import json

            data = {"type": data_type, "values": [1, 2, 3], "metadata": {"created": "2024-01-01"}}
            return json.dumps(data, indent=2)

        prompt_manager.add(complex_data_prompt)

        result = await prompt_manager.get("complex_data_prompt", {"data_type": "test"})

        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        # Should contain the JSON data
        assert isinstance(result.messages[0].content, types.TextContent)
        content_text = result.messages[0].content.text
        assert "test" in content_text
        assert "1" in content_text and "2" in content_text and "3" in content_text
        assert "2024-01-01" in content_text

    @pytest.mark.asyncio
    async def test_get_prompt_function_raises_exception(self, prompt_manager: PromptManager):
        """Test that exceptions in prompt functions are properly handled."""

        def failing_prompt(should_fail: str) -> str:
            """A prompt that fails."""
            if should_fail == "yes":
                raise RuntimeError("Intentional failure")
            return "Success"

        prompt_manager.add(failing_prompt)

        with pytest.raises(ValueError, match="Error getting prompt failing_prompt"):
            await prompt_manager.get("failing_prompt", {"should_fail": "yes"})

    def test_prompt_options_typed_dict(self):
        """Test PromptDefinition TypedDict structure."""
        # This tests the type structure - mainly for documentation
        options: PromptDefinition = {
            "name": "test_name",
            "title": "Test Title",
            "description": "test_description",
            "meta": {"version": "1.0"},
        }

        assert options["name"] == "test_name"
        assert options["title"] == "Test Title"
        assert options["description"] == "test_description"
        assert options["meta"] == {"version": "1.0"}

    def test_decorator_usage(self, prompt_manager: PromptManager):
        """Test using PromptManager as a decorator."""

        @prompt_manager(name="decorated_prompt", title="Decorated")
        def decorated_function(message: str) -> str:
            """A decorated prompt function."""
            return f"Decorated: {message}"

        # Verify the prompt was added
        assert "decorated_prompt" in prompt_manager._prompts
        prompt, func = prompt_manager._prompts["decorated_prompt"]
        assert prompt.name == "decorated_prompt"
        assert prompt.title == "Decorated"
        assert prompt.description == "A decorated prompt function."

    @pytest.mark.asyncio
    async def test_full_workflow(self, prompt_manager: PromptManager):
        """Test a complete workflow: add, list, get, remove."""

        def story_prompt(genre: str, character: str, setting: str = "modern day") -> str:
            """Generate a story prompt."""
            return f"Write a {genre} story about {character} set in {setting}"

        # Add prompt
        added_prompt = prompt_manager.add(story_prompt, title="Story Generator")
        assert added_prompt.name == "story_prompt"
        assert added_prompt.title == "Story Generator"

        # List prompts
        prompts = prompt_manager.list()
        assert len(prompts) == 1
        assert prompts[0] == added_prompt

        # Get prompt
        result = await prompt_manager.get("story_prompt", {"genre": "sci-fi", "character": "a robot"})
        assert isinstance(result.messages[0].content, types.TextContent)
        assert result.messages[0].content.text == "Write a sci-fi story about a robot set in modern day"

        result = await prompt_manager.get(
            "story_prompt", {"genre": "fantasy", "character": "a wizard", "setting": "medieval times"}
        )
        assert isinstance(result.messages[0].content, types.TextContent)
        assert result.messages[0].content.text == "Write a fantasy story about a wizard set in medieval times"

        # Remove prompt
        removed_prompt = prompt_manager.remove("story_prompt")
        assert removed_prompt == added_prompt

        # Verify it's gone
        prompts = prompt_manager.list()
        assert len(prompts) == 0

        # Getting removed prompt should fail
        with pytest.raises(ValueError, match="Prompt story_prompt not found"):
            await prompt_manager.get("story_prompt", {"genre": "mystery", "character": "detective"})

    def test_convert_result_edge_cases(self, prompt_manager: PromptManager):
        """Test _convert_result method with various edge cases."""

        # Test with single string
        result = prompt_manager._convert_result("Hello world")
        assert len(result) == 1
        assert result[0].role == "user"
        assert isinstance(result[0].content, types.TextContent)
        assert result[0].content.text == "Hello world"

        # Test with tuple
        result = prompt_manager._convert_result(("Message 1", "Message 2"))
        assert len(result) == 2
        assert isinstance(result[0].content, types.TextContent)
        assert result[0].content.text == "Message 1"
        assert isinstance(result[1].content, types.TextContent)
        assert result[1].content.text == "Message 2"

        # Test with mixed types
        mixed_input = [
            "String message",
            {"role": "assistant", "content": {"type": "text", "text": "Assistant message"}},
            types.PromptMessage(role="user", content=types.TextContent(type="text", text="User message")),
        ]
        result = prompt_manager._convert_result(mixed_input)
        assert len(result) == 3
        assert isinstance(result[0].content, types.TextContent)
        assert result[0].content.text == "String message"
        assert result[1].role == "assistant"
        assert isinstance(result[1].content, types.TextContent)
        assert result[2].role == "user"
        assert isinstance(result[2].content, types.TextContent)

    def test_convert_result_invalid_data_raises_error(self, prompt_manager: PromptManager):
        """Test that _convert_result raises ValueError for invalid data."""

        # Mock the validation to fail
        invalid_dict = {"invalid": "structure"}

        with pytest.raises(ValueError, match="Could not convert prompt result to message"):
            prompt_manager._convert_result([invalid_dict])
