import asyncio
from typing import Any
from unittest.mock import Mock

import mcp.types as types
import pytest
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.lowlevel.server import Server

from minimcp.server.managers.resource_manager import ResourceDefinition, ResourceManager, _uri_to_pattern


class TestResourceManager:
    """Test suite for ResourceManager class."""

    @pytest.fixture
    def mock_core(self) -> Mock:
        """Create a mock Server for testing."""
        core = Mock(spec=Server)
        core.list_resources = Mock(return_value=Mock())
        core.list_resource_templates = Mock(return_value=Mock())
        core.read_resource = Mock(return_value=Mock())
        return core

    @pytest.fixture
    def resource_manager(self, mock_core: Mock) -> ResourceManager:
        """Create a ResourceManager instance with mocked core."""
        return ResourceManager(mock_core)

    def test_init_hooks_core_methods(self, mock_core: Mock):
        """Test that ResourceManager properly hooks into Server methods."""
        resource_manager = ResourceManager(mock_core)

        # Verify that core methods were called to register handlers
        mock_core.list_resources.assert_called_once()
        mock_core.list_resource_templates.assert_called_once()
        mock_core.read_resource.assert_called_once()

        # Verify internal state
        assert resource_manager._resources == {}

    def test_add_basic_resource(self, resource_manager: ResourceManager):
        """Test adding a basic resource without parameters."""

        def sample_resource() -> str:
            """A sample resource for testing."""
            return "Sample content"

        result = resource_manager.add(sample_resource, "file://test.txt")

        # Verify the returned resource
        assert isinstance(result, types.Resource)
        assert result.name == "sample_resource"
        assert str(result.uri) == "file://test.txt/"
        assert result.description == "A sample resource for testing."
        assert result.title is None
        assert result.mimeType is None
        assert result.annotations is None
        assert result.meta is None

        # Verify internal state
        assert "sample_resource" in resource_manager._resources
        resource_details = resource_manager._resources["sample_resource"]
        assert resource_details.resource == result
        assert resource_details.func == sample_resource
        assert resource_details.normalized_uri == "file://test.txt"
        assert resource_details.uri_pattern is None

    def test_add_resource_template(self, resource_manager: ResourceManager):
        """Test adding a resource template with parameters."""

        def user_profile(user_id: str) -> dict[str, Any]:
            """Get user profile data."""
            return {"id": user_id, "name": f"User {user_id}"}

        result = resource_manager.add(user_profile, "users/{user_id}/profile")

        # Verify the returned resource template
        assert isinstance(result, types.ResourceTemplate)
        assert result.name == "user_profile"
        assert result.uriTemplate == "users/{user_id}/profile"
        assert result.description == "Get user profile data."

        # Verify internal state
        assert "user_profile" in resource_manager._resources
        resource_details = resource_manager._resources["user_profile"]
        assert resource_details.resource == result
        assert resource_details.normalized_uri == "users/|/profile"
        assert resource_details.uri_pattern is not None

    def test_add_resource_with_custom_options(self, resource_manager: ResourceManager):
        """Test adding a resource with custom options."""

        def json_data() -> dict:
            return {"data": "value"}

        custom_annotations = types.Annotations(audience=["user"])
        custom_meta = {"version": "1.0"}

        result = resource_manager.add(
            json_data,
            "data://json",
            title="JSON Data",
            description="Custom JSON data",
            mime_type="application/json",
            annotations=custom_annotations,
            meta=custom_meta,
        )

        assert result.title == "JSON Data"
        assert result.description == "Custom JSON data"
        assert result.mimeType == "application/json"
        assert result.annotations == custom_annotations
        assert result.meta == custom_meta

    def test_add_resource_without_docstring(self, resource_manager: ResourceManager):
        """Test adding a resource without docstring uses None as description."""

        def no_doc_resource() -> str:
            return "content"

        result = resource_manager.add(no_doc_resource, "file://nodoc.txt")
        assert result.description is None

    def test_add_async_resource(self, resource_manager: ResourceManager):
        """Test adding an async function as a resource."""

        async def async_resource(delay: float) -> str:
            """An async resource."""
            await asyncio.sleep(delay)
            return "async content"

        result = resource_manager.add(async_resource, "async/{delay}")

        assert isinstance(result, types.ResourceTemplate)
        assert result.name == "async_resource"
        assert result.description == "An async resource."
        assert "async_resource" in resource_manager._resources

    def test_add_resource_empty_uri_raises_error(self, resource_manager: ResourceManager):
        """Test that adding a resource with empty URI raises ValueError."""

        def test_resource() -> str:
            return "content"

        with pytest.raises(ValueError, match="URI is required"):
            resource_manager.add(test_resource, "")

    def test_add_duplicate_resource_name_raises_error(self, resource_manager: ResourceManager):
        """Test that adding a resource with duplicate name raises ValueError."""

        def resource1() -> str:
            return "content1"

        def resource2() -> str:
            return "content2"

        # Add first resource
        resource_manager.add(resource1, "file://test1.txt", name="duplicate_name")  # type: ignore

        # Adding second resource with same name should raise error
        with pytest.raises(ValueError, match="Resource duplicate_name already registered"):
            resource_manager.add(resource2, "file://test2.txt", name="duplicate_name")  # type: ignore

    def test_add_duplicate_uri_raises_error(self, resource_manager: ResourceManager):
        """Test that adding resources with duplicate normalized URI raises ValueError."""

        def resource1() -> str:
            return "content1"

        def resource2() -> str:
            return "content2"

        # Add first resource
        resource_manager.add(resource1, "file://test.txt")

        # Adding second resource with same URI should raise error
        with pytest.raises(ValueError, match="Resource file://test.txt already registered"):
            resource_manager.add(resource2, "file://test.txt")

    def test_add_duplicate_template_uri_raises_error(self, resource_manager: ResourceManager):
        """Test that adding resource templates with duplicate normalized URI raises ValueError."""

        def template1(id: str) -> str:
            return f"content1-{id}"

        def template2(id: str) -> str:
            return f"content2-{id}"

        # Add first template
        resource_manager.add(template1, "files/{id}")

        # Adding second template with same normalized URI should raise error
        with pytest.raises(ValueError, match="Resource files/\\{id\\} already registered"):
            resource_manager.add(template2, "files/{id}")

    def test_add_template_parameter_mismatch_raises_error(self, resource_manager: ResourceManager):
        """Test that parameter mismatch between URI and function raises ValueError."""

        def mismatched_func(user_id: str, extra_param: str) -> str:
            return f"{user_id}-{extra_param}"

        with pytest.raises(ValueError, match="Mismatch between URI parameters"):
            resource_manager.add(mismatched_func, "users/{user_id}")

        def another_mismatch(wrong_param: str) -> str:
            return wrong_param

        with pytest.raises(ValueError, match="Mismatch between URI parameters"):
            resource_manager.add(another_mismatch, "users/{user_id}")

    def test_remove_existing_resource(self, resource_manager: ResourceManager):
        """Test removing an existing resource."""

        def test_resource() -> str:
            return "content"

        # Add resource first
        added_resource = resource_manager.add(test_resource, "file://test.txt")
        assert "test_resource" in resource_manager._resources

        # Remove the resource
        removed_resource = resource_manager.remove("test_resource")

        assert removed_resource == added_resource
        assert "test_resource" not in resource_manager._resources

    def test_remove_nonexistent_resource_raises_error(self, resource_manager: ResourceManager):
        """Test that removing a non-existent resource raises ValueError."""
        with pytest.raises(ValueError, match="Resource nonexistent not found"):
            resource_manager.remove("nonexistent")

    @pytest.mark.asyncio
    async def test_list_resources_empty(self, resource_manager: ResourceManager):
        """Test listing resources when no resources are registered."""
        result = resource_manager.list()
        assert result == []

        # Test async version
        async_result = await resource_manager._async_list()
        assert async_result == []

    @pytest.mark.asyncio
    async def test_list_resources_with_multiple_resources(self, resource_manager: ResourceManager):
        """Test listing resources when multiple resources are registered."""

        def resource1() -> str:
            return "content1"

        def resource2() -> str:
            return "content2"

        def template1(id: str) -> str:
            return f"template-{id}"

        added_resource1 = resource_manager.add(resource1, "file://test1.txt")
        added_resource2 = resource_manager.add(resource2, "file://test2.txt")
        # Template should not appear in regular list
        resource_manager.add(template1, "templates/{id}")

        result = resource_manager.list()

        assert len(result) == 2
        assert added_resource1 in result
        assert added_resource2 in result

        # Test async version
        async_result = await resource_manager._async_list()
        assert len(async_result) == 2
        assert added_resource1 in async_result
        assert added_resource2 in async_result

    @pytest.mark.asyncio
    async def test_list_resource_templates_empty(self, resource_manager: ResourceManager):
        """Test listing resource templates when no templates are registered."""
        result = resource_manager.list_templates()
        assert result == []

        # Test async version
        async_result = await resource_manager._async_list_templates()
        assert async_result == []

    @pytest.mark.asyncio
    async def test_list_resource_templates_with_multiple_templates(self, resource_manager: ResourceManager):
        """Test listing resource templates when multiple templates are registered."""

        def template1(id: str) -> str:
            return f"template1-{id}"

        def template2(name: str) -> str:
            return f"template2-{name}"

        def regular_resource() -> str:
            return "regular content"

        added_template1 = resource_manager.add(template1, "templates/{id}")
        added_template2 = resource_manager.add(template2, "items/{name}")
        # Regular resource should not appear in template list
        resource_manager.add(regular_resource, "file://regular.txt")

        result = resource_manager.list_templates()

        assert len(result) == 2
        assert added_template1 in result
        assert added_template2 in result

        # Test async version
        async_result = await resource_manager._async_list_templates()
        assert len(async_result) == 2
        assert added_template1 in async_result
        assert added_template2 in async_result

    @pytest.mark.asyncio
    async def test_read_resource_sync_function(self, resource_manager: ResourceManager):
        """Test reading a synchronous resource."""

        def text_resource() -> str:
            """A text resource."""
            return "Hello, World!"

        resource_manager.add(text_resource, "file://hello.txt", mime_type="text/plain")

        result = await resource_manager.read("file://hello.txt")
        result_list = list(result)

        assert len(result_list) == 1
        assert isinstance(result_list[0], ReadResourceContents)
        assert result_list[0].content == "Hello, World!"
        assert result_list[0].mime_type == "text/plain"

    @pytest.mark.asyncio
    async def test_read_resource_async_function(self, resource_manager: ResourceManager):
        """Test reading an asynchronous resource."""

        async def async_text_resource() -> str:
            """An async text resource."""
            await asyncio.sleep(0.01)  # Small delay to make it actually async
            return "Async Hello, World!"

        resource_manager.add(async_text_resource, "file://async_hello.txt")

        result = await resource_manager.read("file://async_hello.txt")
        result_list = list(result)

        assert len(result_list) == 1
        assert result_list[0].content == "Async Hello, World!"

    @pytest.mark.asyncio
    async def test_read_resource_template_with_parameters(self, resource_manager: ResourceManager):
        """Test reading a resource template with parameters."""

        def user_data(user_id: str, format: str) -> dict[str, Any]:
            """Get user data in specified format."""
            return {"id": user_id, "format": format, "data": f"User {user_id} data"}

        resource_manager.add(user_data, "users/{user_id}/{format}")

        result = await resource_manager.read("users/123/json")
        result_list = list(result)

        assert len(result_list) == 1
        content = result_list[0].content
        # Should be JSON serialized
        content_str = content if isinstance(content, str) else content.decode()
        assert "123" in content_str
        assert "json" in content_str
        assert "User 123 data" in content_str

    @pytest.mark.asyncio
    async def test_read_resource_bytes_content(self, resource_manager: ResourceManager):
        """Test reading a resource that returns bytes."""

        def binary_resource() -> bytes:
            """A binary resource."""
            return b"Binary content"

        resource_manager.add(binary_resource, "file://binary.dat", mime_type="application/octet-stream")

        result = await resource_manager.read("file://binary.dat")
        result_list = list(result)

        assert len(result_list) == 1
        assert result_list[0].content == b"Binary content"
        assert result_list[0].mime_type == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_read_resource_complex_data(self, resource_manager: ResourceManager):
        """Test reading a resource that returns complex data structures."""

        def json_resource() -> dict[str, Any]:
            """A JSON resource."""
            return {"name": "Test Resource", "values": [1, 2, 3], "metadata": {"created": "2024-01-01"}}

        resource_manager.add(json_resource, "data://json", mime_type="application/json")

        result = await resource_manager.read("data://json")
        result_list = list(result)

        assert len(result_list) == 1
        content = result_list[0].content
        content_str = content if isinstance(content, str) else content.decode()
        assert "Test Resource" in content_str
        assert result_list[0].mime_type == "application/json"

    @pytest.mark.asyncio
    async def test_read_nonexistent_resource_raises_error(self, resource_manager: ResourceManager):
        """Test that reading a non-existent resource raises ValueError."""
        with pytest.raises(ValueError, match="Resource file://nonexistent.txt not found"):
            await resource_manager.read("file://nonexistent.txt")

    @pytest.mark.asyncio
    async def test_read_by_name_existing_resource(self, resource_manager: ResourceManager):
        """Test reading a resource by name."""

        def named_resource() -> str:
            return "Named content"

        resource_manager.add(named_resource, "file://named.txt")

        result = await resource_manager.read_by_name("named_resource")
        result_list = list(result)

        assert len(result_list) == 1
        assert result_list[0].content == "Named content"

    @pytest.mark.asyncio
    async def test_read_by_name_template_with_args(self, resource_manager: ResourceManager):
        """Test reading a resource template by name with arguments."""

        def template_resource(item_id: str, category: str) -> str:
            return f"Item {item_id} in category {category}"

        resource_manager.add(template_resource, "items/{item_id}/{category}")

        result = await resource_manager.read_by_name("template_resource", {"item_id": "123", "category": "books"})
        result_list = list(result)

        assert len(result_list) == 1
        assert result_list[0].content == "Item 123 in category books"

    @pytest.mark.asyncio
    async def test_read_by_name_nonexistent_raises_error(self, resource_manager: ResourceManager):
        """Test that reading a non-existent resource by name raises ValueError."""
        with pytest.raises(ValueError, match="Resource nonexistent not found"):
            await resource_manager.read_by_name("nonexistent")

    @pytest.mark.asyncio
    async def test_read_resource_function_raises_exception(self, resource_manager: ResourceManager):
        """Test that exceptions in resource functions are properly handled."""

        def failing_resource() -> str:
            """A resource that fails."""
            raise RuntimeError("Intentional failure")

        resource_manager.add(failing_resource, "file://failing.txt")

        with pytest.raises(ValueError, match="Error reading resource failing_resource"):
            await resource_manager.read("file://failing.txt")

    def test_resource_options_typed_dict(self):
        """Test ResourceDefinition TypedDict structure."""
        # This tests the type structure - mainly for documentation
        options: ResourceDefinition = {
            "title": "Test Title",
            "description": "test_description",
            "mime_type": "text/plain",
            "annotations": types.Annotations(audience=["user"]),
            "meta": {"version": "1.0"},
        }

        assert options["title"] == "Test Title"
        assert options["description"] == "test_description"
        assert options["mime_type"] == "text/plain"
        assert options["meta"] == {"version": "1.0"}

    def test_decorator_usage(self, resource_manager: ResourceManager):
        """Test using ResourceManager as a decorator."""

        @resource_manager("file://decorated.txt", title="Decorated Resource")
        def decorated_function() -> str:
            """A decorated resource function."""
            return "Decorated content"

        # Verify the resource was added
        assert "decorated_function" in resource_manager._resources
        resource_details = resource_manager._resources["decorated_function"]
        assert resource_details.resource.name == "decorated_function"
        assert resource_details.resource.title == "Decorated Resource"
        assert resource_details.resource.description == "A decorated resource function."

    def test_uri_to_pattern_function(self):
        """Test the _uri_to_pattern utility function."""

        # Simple template
        pattern = _uri_to_pattern("users/{id}")
        assert pattern.match("users/123")
        match = pattern.match("users/abc")
        assert match is not None
        assert match.groupdict() == {"id": "abc"}
        assert not pattern.match("users/123/extra")

        # Multiple parameters
        pattern = _uri_to_pattern("users/{user_id}/posts/{post_id}")
        match = pattern.match("users/123/posts/456")
        assert match is not None
        assert match.groupdict() == {"user_id": "123", "post_id": "456"}

        # With special characters
        pattern = _uri_to_pattern("files/{name}.{ext}")
        match = pattern.match("files/document.pdf")
        assert match is not None
        assert match.groupdict() == {"name": "document", "ext": "pdf"}

        # No match cases
        pattern = _uri_to_pattern("users/{id}")
        assert not pattern.match("posts/123")
        assert not pattern.match("users/")
        assert not pattern.match("users/123/")

    def test_get_normalized_uri(self, resource_manager: ResourceManager):
        """Test the _get_normalized_uri method."""

        assert resource_manager._get_normalized_uri("users/{id}") == "users/|"
        assert resource_manager._get_normalized_uri("users/{user_id}/posts/{post_id}") == "users/|/posts/|"
        assert resource_manager._get_normalized_uri("files/static.txt") == "files/static.txt"
        assert resource_manager._get_normalized_uri("complex/{a}/path/{b}/file.{ext}") == "complex/|/path/|/file.|"

    def test_find_matching_details_exact_match(self, resource_manager: ResourceManager):
        """Test _find_matching_details with exact URI match."""

        def static_resource() -> str:
            return "static content"

        resource_manager.add(static_resource, "file://static.txt")

        details, args = resource_manager._find_matching_details("file://static.txt")
        assert details is not None
        assert args is None
        assert details.resource.name == "static_resource"

    def test_find_matching_details_template_match(self, resource_manager: ResourceManager):
        """Test _find_matching_details with template URI match."""

        def template_resource(id: str, format: str) -> str:
            return f"content-{id}-{format}"

        resource_manager.add(template_resource, "items/{id}.{format}")

        details, args = resource_manager._find_matching_details("items/123.json")
        assert details is not None
        assert args == {"id": "123", "format": "json"}
        assert details.resource.name == "template_resource"

    def test_find_matching_details_no_match(self, resource_manager: ResourceManager):
        """Test _find_matching_details with no matching URI."""

        def some_resource() -> str:
            return "content"

        resource_manager.add(some_resource, "file://test.txt")

        details, args = resource_manager._find_matching_details("file://nonexistent.txt")
        assert details is None
        assert args is None

    @pytest.mark.asyncio
    async def test_full_workflow(self, resource_manager: ResourceManager):
        """Test a complete workflow: add, list, read, remove."""

        def config_resource() -> dict[str, Any]:
            """Application configuration."""
            return {"app_name": "TestApp", "version": "1.0", "debug": True}

        def user_resource(user_id: str) -> dict[str, Any]:
            """User profile data."""
            return {"id": user_id, "name": f"User {user_id}", "active": True}

        # Add resources
        config_added = resource_manager.add(
            config_resource, "config://app.json", title="App Config", mime_type="application/json"
        )
        user_added = resource_manager.add(user_resource, "users/{user_id}")

        assert config_added.name == "config_resource"
        assert isinstance(user_added, types.ResourceTemplate)

        # List resources and templates
        resources = resource_manager.list()
        templates = resource_manager.list_templates()

        assert len(resources) == 1
        assert len(templates) == 1
        assert config_added in resources
        assert user_added in templates

        # Read resources
        config_result = await resource_manager.read("config://app.json")
        config_content = list(config_result)[0]
        config_content_str = (
            config_content.content if isinstance(config_content.content, str) else config_content.content.decode()
        )
        assert "TestApp" in config_content_str
        assert config_content.mime_type == "application/json"

        user_result = await resource_manager.read("users/456")
        user_content = list(user_result)[0]
        user_content_str = (
            user_content.content if isinstance(user_content.content, str) else user_content.content.decode()
        )
        assert "User 456" in user_content_str

        # Read by name
        user_by_name = await resource_manager.read_by_name("user_resource", {"user_id": "789"})
        user_by_name_content = list(user_by_name)[0]
        user_by_name_content_str = (
            user_by_name_content.content
            if isinstance(user_by_name_content.content, str)
            else user_by_name_content.content.decode()
        )
        assert "User 789" in user_by_name_content_str

        # Remove resources
        removed_config = resource_manager.remove("config_resource")
        removed_user = resource_manager.remove("user_resource")

        assert removed_config == config_added
        assert removed_user == user_added

        # Verify they're gone
        assert len(resource_manager.list()) == 0
        assert len(resource_manager.list_templates()) == 0

        # Reading removed resources should fail
        with pytest.raises(ValueError, match="Resource config://app.json not found"):
            await resource_manager.read("config://app.json")

        with pytest.raises(ValueError, match="Resource user_resource not found"):
            await resource_manager.read_by_name("user_resource", {"user_id": "123"})
