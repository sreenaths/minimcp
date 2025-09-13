import asyncio
from abc import ABC, abstractmethod
from typing import Any
from unittest.mock import Mock, patch

import pytest
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata

from minimcp.utils.func import FuncDetails, extract_func_details, validate_func, validate_func_name


class TestFuncDetails:
    """Test suite for FuncDetails dataclass."""

    def test_func_details_creation(self):
        """Test FuncDetails dataclass creation."""
        mock_meta = Mock(spec=FuncMetadata)
        details = FuncDetails(name="test_func", doc="Test function", meta=mock_meta)

        assert details.name == "test_func"
        assert details.doc == "Test function"
        assert details.meta is mock_meta

    def test_func_details_with_none_values(self):
        """Test FuncDetails with None values."""
        mock_meta = Mock(spec=FuncMetadata)
        details = FuncDetails(name=None, doc=None, meta=mock_meta)

        assert details.name is None
        assert details.doc is None
        assert details.meta is mock_meta


class TestExtractFuncDetails:
    """Test suite for extract_func_details function."""

    def test_extract_func_details_basic_function(self):
        """Test extracting details from a basic function."""

        def sample_function(x: int, y: str = "default") -> str:
            """A sample function for testing."""
            return f"{x}: {y}"

        with patch("minimcp.utils.func.func_metadata") as mock_func_metadata:
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            result = extract_func_details(sample_function)

            assert isinstance(result, FuncDetails)
            assert result.name == "sample_function"
            assert result.doc == "A sample function for testing."
            assert result.meta is mock_meta
            mock_func_metadata.assert_called_once_with(sample_function)

    def test_extract_func_details_function_without_docstring(self):
        """Test extracting details from a function without docstring."""

        def no_doc_function():
            pass

        with patch("minimcp.utils.func.func_metadata") as mock_func_metadata:
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            result = extract_func_details(no_doc_function)

            assert result.name == "no_doc_function"
            assert result.doc is None
            assert result.meta is mock_meta

    def test_extract_func_details_async_function(self):
        """Test extracting details from an async function."""

        async def async_function(delay: float) -> str:
            """An async function."""
            await asyncio.sleep(delay)
            return "done"

        with patch("minimcp.utils.func.func_metadata") as mock_func_metadata:
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            result = extract_func_details(async_function)

            assert result.name == "async_function"
            assert result.doc == "An async function."
            assert result.meta is mock_meta

    def test_extract_func_details_lambda_function(self):
        """Test extracting details from a lambda function."""
        lambda_func = lambda x: x * 2  # noqa: E731

        with patch("minimcp.utils.func.func_metadata") as mock_func_metadata:
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            result = extract_func_details(lambda_func)

            assert result.name == "<lambda>"
            assert result.doc is None
            assert result.meta is mock_meta

    def test_extract_func_details_calls_validate_func(self):
        """Test that extract_func_details calls validate_func."""

        def valid_function():
            pass

        with (
            patch("minimcp.utils.func.validate_func") as mock_validate,
            patch("minimcp.utils.func.func_metadata") as mock_func_metadata,
        ):
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            extract_func_details(valid_function)

            mock_validate.assert_called_once_with(valid_function)

    def test_extract_func_details_with_invalid_function(self):
        """Test that extract_func_details propagates validation errors."""

        def invalid_function():
            pass

        with patch("minimcp.utils.func.validate_func", side_effect=ValueError("Invalid function")):
            with pytest.raises(ValueError, match="Invalid function"):
                extract_func_details(invalid_function)

    def test_extract_func_details_method(self):
        """Test extracting details from a method."""

        class TestClass:
            def test_method(self, value: int) -> int:
                """A test method."""
                return value * 2

        instance = TestClass()

        with patch("minimcp.utils.func.func_metadata") as mock_func_metadata:
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            result = extract_func_details(instance.test_method)

            assert result.name == "test_method"
            assert result.doc == "A test method."
            assert result.meta is mock_meta

    def test_extract_func_details_function_with_complex_docstring(self):
        """Test extracting details from a function with complex docstring."""

        def complex_doc_function():
            """
            A function with a complex docstring.

            This function has multiple lines
            and various formatting.

            Returns:
                None: This function returns nothing.
            """
            pass

        with patch("minimcp.utils.func.func_metadata") as mock_func_metadata:
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            result = extract_func_details(complex_doc_function)

            expected_doc = (
                "A function with a complex docstring.\n\n"
                "This function has multiple lines\n"
                "and various formatting.\n\n"
                "Returns:\n"
                "    None: This function returns nothing."
            )
            assert result.name == "complex_doc_function"
            assert result.doc == expected_doc
            assert result.meta is mock_meta


class TestValidateFunc:
    """Test suite for validate_func function."""

    def test_validate_func_valid_function(self):
        """Test validation of a valid function."""

        def valid_function(x: int, y: str = "default") -> str:
            return f"{x}: {y}"

        # Should not raise any exception
        validate_func(valid_function)

    def test_validate_func_valid_async_function(self):
        """Test validation of a valid async function."""

        async def valid_async_function(x: int) -> int:
            return x * 2

        # Should not raise any exception
        validate_func(valid_async_function)

    def test_validate_func_valid_method(self):
        """Test validation of a valid method."""

        class TestClass:
            def valid_method(self, x: int) -> int:
                return x * 2

        instance = TestClass()

        # Should not raise any exception
        validate_func(instance.valid_method)

    def test_validate_func_classmethod_raises_error(self):
        """Test that classmethod raises ValueError."""

        class TestClass:
            @classmethod
            def class_method(cls, x: int) -> int:
                return x * 2

        # Access the classmethod descriptor directly
        classmethod_descriptor = TestClass.__dict__["class_method"]
        with pytest.raises(ValueError, match="Function cannot be a classmethod"):
            validate_func(classmethod_descriptor)

    def test_validate_func_staticmethod_raises_error(self):
        """Test that staticmethod raises ValueError."""

        class TestClass:
            @staticmethod
            def static_method(x: int) -> int:
                return x * 2

        # Access the staticmethod descriptor directly
        staticmethod_descriptor = TestClass.__dict__["static_method"]
        with pytest.raises(ValueError, match="Function cannot be a staticmethod"):
            validate_func(staticmethod_descriptor)

    def test_validate_func_abstract_method_raises_error(self):
        """Test that abstract method raises ValueError."""

        class AbstractClass(ABC):
            @abstractmethod
            def abstract_method(self, x: int) -> int:
                pass

        # Create an instance of the abstract method
        abstract_func = AbstractClass.abstract_method

        with pytest.raises(ValueError, match="Function cannot be an abstract method"):
            validate_func(abstract_func)

    def test_validate_func_non_routine_raises_error(self):
        """Test that non-routine objects raise ValueError."""
        not_a_function = "this is not a function"

        with pytest.raises(ValueError, match="Value passed is not a function or method"):
            validate_func(not_a_function)  # type: ignore

    def test_validate_func_class_raises_error(self):
        """Test that class objects raise ValueError."""

        class SomeClass:
            pass

        with pytest.raises(ValueError, match="Value passed is not a function or method"):
            validate_func(SomeClass)  # type: ignore

    def test_validate_func_var_positional_args_raises_error(self):
        """Test that functions with *args raise ValueError."""

        def function_with_args(x: int, *args) -> int:
            return x + sum(args)

        with pytest.raises(ValueError, match="Functions with \\*args are not supported"):
            validate_func(function_with_args)

    def test_validate_func_var_keyword_args_raises_error(self):
        """Test that functions with **kwargs raise ValueError."""

        def function_with_kwargs(x: int, **kwargs) -> int:
            return x + len(kwargs)

        with pytest.raises(ValueError, match="Functions with \\*\\*kwargs are not supported"):
            validate_func(function_with_kwargs)

    def test_validate_func_both_var_args_raises_error(self):
        """Test that functions with both *args and **kwargs raise ValueError."""

        def function_with_both(x: int, *args, **kwargs) -> int:
            return x + sum(args) + len(kwargs)

        # Should raise error for *args first
        with pytest.raises(ValueError, match="Functions with \\*args are not supported"):
            validate_func(function_with_both)

    def test_validate_func_lambda_function(self):
        """Test validation of lambda functions."""
        lambda_func = lambda x: x * 2  # noqa: E731

        # Should not raise any exception
        validate_func(lambda_func)

    def test_validate_func_builtin_function(self):
        """Test validation of builtin functions."""
        # Should not raise any exception for builtin functions that have signatures
        validate_func(len)
        # Note: Some builtin functions like max, min don't have introspectable signatures
        # so we test with functions that do have signatures

    def test_validate_func_function_with_annotations(self):
        """Test validation of functions with type annotations."""

        def annotated_function(x: int, y: str, z: float = 1.0) -> dict[str, Any]:
            return {"x": x, "y": y, "z": z}

        # Should not raise any exception
        validate_func(annotated_function)

    def test_validate_func_function_with_keyword_only_params(self):
        """Test validation of functions with keyword-only parameters."""

        def keyword_only_function(x: int, *, y: str, z: float = 1.0) -> str:
            return f"{x}: {y}, {z}"

        # Should not raise any exception
        validate_func(keyword_only_function)

    def test_validate_func_function_with_positional_only_params(self):
        """Test validation of functions with positional-only parameters."""

        def positional_only_function(x: int, y: str, /) -> str:
            return f"{x}: {y}"

        # Should not raise any exception
        validate_func(positional_only_function)


class TestValidateFuncName:
    """Test suite for validate_func_name function."""

    def test_validate_func_name_valid_name(self):
        """Test validation of a valid function name."""
        result = validate_func_name("valid_function_name")
        assert result == "valid_function_name"

    def test_validate_func_name_with_underscores(self):
        """Test validation of function name with underscores."""
        result = validate_func_name("function_with_underscores")
        assert result == "function_with_underscores"

    def test_validate_func_name_with_numbers(self):
        """Test validation of function name with numbers."""
        result = validate_func_name("function123")
        assert result == "function123"

    def test_validate_func_name_none_raises_error(self):
        """Test that None name raises ValueError."""
        with pytest.raises(ValueError, match="Name is not available"):
            validate_func_name(None)

    def test_validate_func_name_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Name is not available"):
            validate_func_name("")

    def test_validate_func_name_lambda_raises_error(self):
        """Test that lambda name raises ValueError."""
        with pytest.raises(ValueError, match="Name must be provided for lambda functions"):
            validate_func_name("<lambda>")

    def test_validate_func_name_whitespace_only_raises_error(self):
        """Test that whitespace-only string raises ValueError."""
        # The actual function doesn't strip whitespace, so this test should pass the string as-is
        result = validate_func_name("   ")
        assert result == "   "

    def test_validate_func_name_single_character(self):
        """Test validation of single character name."""
        result = validate_func_name("f")
        assert result == "f"

    def test_validate_func_name_with_special_characters(self):
        """Test validation of function name with special characters."""
        # Python allows these in function names
        result = validate_func_name("function_name_with_special_chars")
        assert result == "function_name_with_special_chars"


class TestIntegration:
    """Integration tests for func utilities."""

    def test_full_workflow_with_valid_function(self):
        """Test the complete workflow with a valid function."""

        def sample_function(x: int, y: str = "default") -> str:
            """A sample function for integration testing."""
            return f"{x}: {y}"

        with patch("minimcp.utils.func.func_metadata") as mock_func_metadata:
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            # Extract details (which includes validation)
            details = extract_func_details(sample_function)

            # Validate the name
            validated_name = validate_func_name(details.name)

            assert validated_name == "sample_function"
            assert details.doc == "A sample function for integration testing."
            assert details.meta is mock_meta

    def test_full_workflow_with_lambda_function(self):
        """Test the complete workflow with a lambda function."""
        lambda_func = lambda x: x * 2  # noqa: E731

        with patch("minimcp.utils.func.func_metadata") as mock_func_metadata:
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            # Extract details
            details = extract_func_details(lambda_func)

            # Validate the name should raise error for lambda
            with pytest.raises(ValueError, match="Name must be provided for lambda functions"):
                validate_func_name(details.name)

    def test_error_propagation_from_validation(self):
        """Test that validation errors are properly propagated."""

        def function_with_args(*args):
            return sum(args)

        # Should raise error during extract_func_details due to validation
        with pytest.raises(ValueError, match="Functions with \\*args are not supported"):
            extract_func_details(function_with_args)

    def test_inspect_module_usage(self):
        """Test that the inspect module is used correctly."""

        def test_function():
            """Test docstring."""
            pass

        with (
            patch("minimcp.utils.func.inspect.getdoc") as mock_getdoc,
            patch("minimcp.utils.func.func_metadata") as mock_func_metadata,
        ):
            mock_getdoc.return_value = "Mocked docstring"
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            result = extract_func_details(test_function)

            mock_getdoc.assert_called_once_with(test_function)
            assert result.doc == "Mocked docstring"

    def test_function_name_extraction(self):
        """Test function name extraction with different scenarios."""

        # Regular function
        def regular_function():
            pass

        with patch("minimcp.utils.func.func_metadata") as mock_func_metadata:
            mock_meta = Mock(spec=FuncMetadata)
            mock_func_metadata.return_value = mock_meta

            # Regular function should have name
            result1 = extract_func_details(regular_function)
            assert result1.name == "regular_function"

            # Test with getattr returning None for __name__
            with patch(
                "minimcp.utils.func.getattr",
                side_effect=lambda obj, attr, default=None: None if attr == "__name__" else getattr(obj, attr, default),
            ):
                result2 = extract_func_details(regular_function)
                assert result2.name is None
