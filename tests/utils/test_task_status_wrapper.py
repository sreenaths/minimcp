from typing import Any
from unittest.mock import Mock

import pytest
from anyio.abc import TaskStatus

from minimcp.utils.task_status_wrapper import TaskStatusWrapper


class TestTaskStatusWrapper:
    """Test suite for TaskStatusWrapper class."""

    @pytest.fixture
    def mock_task_status(self):
        """Create a mock TaskStatus for testing."""
        return Mock(spec=TaskStatus)

    @pytest.fixture
    def wrapper(self, mock_task_status):
        """Create a TaskStatusWrapper instance for testing."""
        return TaskStatusWrapper(mock_task_status)

    def test_init_sets_inner_and_is_set_false(self, mock_task_status):
        """Test that __init__ properly initializes the wrapper."""
        wrapper = TaskStatusWrapper(mock_task_status)

        assert wrapper._inner is mock_task_status
        assert wrapper._is_set is False
        assert wrapper.is_set is False

    def test_init_with_different_types(self):
        """Test initialization with different TaskStatus types."""
        # Test with string type
        mock_status_str = Mock(spec=TaskStatus[str])
        wrapper_str = TaskStatusWrapper(mock_status_str)
        assert wrapper_str._inner is mock_status_str

        # Test with int type
        mock_status_int = Mock(spec=TaskStatus[int])
        wrapper_int = TaskStatusWrapper(mock_status_int)
        assert wrapper_int._inner is mock_status_int

        # Test with Any type
        mock_status_any = Mock(spec=TaskStatus[Any])
        wrapper_any = TaskStatusWrapper(mock_status_any)
        assert wrapper_any._inner is mock_status_any

    def test_set_first_time_returns_true_and_calls_started(self, wrapper, mock_task_status):
        """Test that first call to set() returns True and calls started()."""
        test_value = "test_value"

        result = wrapper.set(test_value)

        assert result is True
        assert wrapper._is_set is True
        assert wrapper.is_set is True
        mock_task_status.started.assert_called_once_with(test_value)

    def test_set_second_time_returns_false_and_no_call(self, wrapper, mock_task_status):
        """Test that second call to set() returns False and doesn't call started()."""
        test_value1 = "first_value"
        test_value2 = "second_value"

        # First call
        result1 = wrapper.set(test_value1)
        assert result1 is True

        # Reset mock to check second call
        mock_task_status.reset_mock()

        # Second call
        result2 = wrapper.set(test_value2)

        assert result2 is False
        assert wrapper._is_set is True  # Still set
        assert wrapper.is_set is True
        mock_task_status.started.assert_not_called()

    def test_set_multiple_times_only_first_succeeds(self, wrapper, mock_task_status):
        """Test that multiple calls to set() only the first one succeeds."""
        values = ["value1", "value2", "value3", "value4"]
        results = []

        for value in values:
            result = wrapper.set(value)
            results.append(result)

        # Only first should return True
        assert results == [True, False, False, False]
        assert wrapper.is_set is True

        # started() should only be called once with first value
        mock_task_status.started.assert_called_once_with("value1")

    def test_set_with_different_value_types(self, mock_task_status):
        """Test set() with different value types."""
        # Test with string
        wrapper_str = TaskStatusWrapper(mock_task_status)
        result_str = wrapper_str.set("string_value")
        assert result_str is True
        mock_task_status.started.assert_called_with("string_value")

        # Reset for next test
        mock_task_status.reset_mock()

        # Test with int
        wrapper_int = TaskStatusWrapper(mock_task_status)
        result_int = wrapper_int.set(42)
        assert result_int is True
        mock_task_status.started.assert_called_with(42)

        # Reset for next test
        mock_task_status.reset_mock()

        # Test with dict
        wrapper_dict = TaskStatusWrapper(mock_task_status)
        test_dict = {"key": "value", "number": 123}
        result_dict = wrapper_dict.set(test_dict)
        assert result_dict is True
        mock_task_status.started.assert_called_with(test_dict)

    def test_set_with_none_value(self, wrapper, mock_task_status):
        """Test set() with None value."""
        result = wrapper.set(None)

        assert result is True
        assert wrapper.is_set is True
        mock_task_status.started.assert_called_once_with(None)

    def test_is_set_property_reflects_state(self, wrapper):
        """Test that is_set property correctly reflects the wrapper state."""
        # Initially not set
        assert wrapper.is_set is False

        # After setting
        wrapper.set("test")
        assert wrapper.is_set is True

        # Remains set after additional calls
        wrapper.set("another_test")
        assert wrapper.is_set is True

    def test_is_set_property_is_read_only(self, wrapper):
        """Test that is_set property cannot be directly modified."""
        # This should not be possible to set directly
        # The property should only reflect the internal _is_set state
        assert wrapper.is_set is False

        # Attempting to set the property should not work
        # (Python properties are read-only unless explicitly made settable)
        with pytest.raises(AttributeError):
            wrapper.is_set = True

    def test_wrapper_preserves_generic_type_information(self):
        """Test that the wrapper preserves generic type information."""
        # This is more of a type checking test, but we can verify behavior
        mock_status_str = Mock(spec=TaskStatus[str])
        wrapper_str = TaskStatusWrapper[str](mock_status_str)

        # Should work with string
        result = wrapper_str.set("string_value")
        assert result is True
        mock_status_str.started.assert_called_once_with("string_value")

    def test_concurrent_set_calls_thread_safety_note(self, wrapper, mock_task_status):
        """Test behavior with rapid successive calls (simulating concurrency issues)."""
        # Note: The comment in the code mentions this is not thread-safe
        # This test verifies the current behavior, not thread safety

        # Rapid successive calls
        results = []
        for i in range(10):
            result = wrapper.set(f"value_{i}")
            results.append(result)

        # Only first should succeed
        assert results[0] is True
        assert all(not result for result in results[1:])

        # Only one call to started
        mock_task_status.started.assert_called_once_with("value_0")

    def test_wrapper_state_consistency(self, wrapper, mock_task_status):
        """Test that wrapper state remains consistent."""
        # Initial state
        assert wrapper._is_set is False
        assert wrapper.is_set is False

        # After first set
        wrapper.set("test_value")
        assert wrapper._is_set is True
        assert wrapper.is_set is True

        # After subsequent sets
        wrapper.set("another_value")
        wrapper.set("yet_another_value")
        assert wrapper._is_set is True
        assert wrapper.is_set is True

    def test_inner_task_status_exception_handling(self, wrapper, mock_task_status):
        """Test behavior when inner TaskStatus raises an exception."""
        # Make started() raise an exception
        mock_task_status.started.side_effect = RuntimeError("Task status error")

        # The exception should propagate
        with pytest.raises(RuntimeError, match="Task status error"):
            wrapper.set("test_value")

        # The wrapper should still be marked as set (since we set _is_set before calling started)
        assert wrapper._is_set is True
        assert wrapper.is_set is True

    def test_inner_task_status_exception_prevents_double_set(self, wrapper, mock_task_status):
        """Test that exception in first set() prevents second set() from working."""
        # Make started() raise an exception
        mock_task_status.started.side_effect = RuntimeError("Task status error")

        # First call should raise exception but mark as set
        with pytest.raises(RuntimeError):
            wrapper.set("first_value")

        # Reset the mock to not raise exception
        mock_task_status.started.side_effect = None

        # Second call should return False (already set)
        result = wrapper.set("second_value")
        assert result is False

        # started should not be called again
        mock_task_status.started.assert_called_once_with("first_value")

    def test_wrapper_with_complex_objects(self, wrapper, mock_task_status):
        """Test wrapper with complex objects as values."""

        class ComplexObject:
            def __init__(self, name: str, data: dict):
                self.name = name
                self.data = data

            def __eq__(self, other):
                return isinstance(other, ComplexObject) and self.name == other.name and self.data == other.data

        complex_obj = ComplexObject("test", {"key": "value", "nested": {"a": 1}})

        result = wrapper.set(complex_obj)

        assert result is True
        assert wrapper.is_set is True
        mock_task_status.started.assert_called_once_with(complex_obj)

    def test_wrapper_docstring_and_comments_behavior(self, wrapper):
        """Test that the wrapper behaves according to its docstring/comments."""
        # The comment mentions "Not using a lock as set is synchronous"
        # This test verifies the synchronous behavior

        # Multiple rapid calls should be handled synchronously
        results = [wrapper.set(f"sync_test_{i}") for i in range(5)]

        # First should succeed, rest should fail
        assert results == [True, False, False, False, False]

    def test_type_var_t_usage(self):
        """Test that TypeVar T is properly used."""
        # Create wrappers with different types
        mock_status_int = Mock(spec=TaskStatus[int])
        mock_status_str = Mock(spec=TaskStatus[str])

        wrapper_int = TaskStatusWrapper(mock_status_int)
        wrapper_str = TaskStatusWrapper(mock_status_str)

        # Both should work with their respective types
        wrapper_int.set(123)
        wrapper_str.set("test")

        mock_status_int.started.assert_called_once_with(123)
        mock_status_str.started.assert_called_once_with("test")


class TestTaskStatusWrapperIntegration:
    """Integration tests for TaskStatusWrapper."""

    def test_integration_with_real_task_status_interface(self):
        """Test integration with objects that implement TaskStatus interface."""

        class MockTaskStatusImplementation:
            def __init__(self):
                self.started_called = False
                self.started_value = None

            def started(self, value):
                self.started_called = True
                self.started_value = value

        mock_impl = MockTaskStatusImplementation()
        wrapper = TaskStatusWrapper(mock_impl)  # type: ignore

        # Test the integration
        result = wrapper.set("integration_test")

        assert result is True
        assert wrapper.is_set is True
        assert mock_impl.started_called is True
        assert mock_impl.started_value == "integration_test"

    def test_wrapper_lifecycle(self):
        """Test the complete lifecycle of a TaskStatusWrapper."""
        mock_status = Mock(spec=TaskStatus)

        # Creation
        wrapper = TaskStatusWrapper(mock_status)
        assert not wrapper.is_set

        # First use
        result1 = wrapper.set("lifecycle_value")
        assert result1 is True
        assert wrapper.is_set

        # Subsequent uses
        result2 = wrapper.set("another_value")
        assert result2 is False
        assert wrapper.is_set

        # Verify final state
        mock_status.started.assert_called_once_with("lifecycle_value")

    def test_multiple_wrappers_independence(self):
        """Test that multiple wrappers are independent."""
        mock_status1 = Mock(spec=TaskStatus)
        mock_status2 = Mock(spec=TaskStatus)

        wrapper1 = TaskStatusWrapper(mock_status1)
        wrapper2 = TaskStatusWrapper(mock_status2)

        # Set first wrapper
        result1 = wrapper1.set("wrapper1_value")
        assert result1 is True
        assert wrapper1.is_set is True
        assert wrapper2.is_set is False

        # Set second wrapper
        result2 = wrapper2.set("wrapper2_value")
        assert result2 is True
        assert wrapper1.is_set is True
        assert wrapper2.is_set is True

        # Verify independent calls
        mock_status1.started.assert_called_once_with("wrapper1_value")
        mock_status2.started.assert_called_once_with("wrapper2_value")

    def test_error_scenarios(self):
        """Test various error scenarios."""
        # Test with None inner status
        with pytest.raises(AttributeError):
            wrapper = TaskStatusWrapper(None)  # type: ignore
            wrapper.set("test")

        # Test with object that doesn't have started method
        class InvalidTaskStatus:
            pass

        invalid_status = InvalidTaskStatus()
        wrapper = TaskStatusWrapper(invalid_status)  # type: ignore

        with pytest.raises(AttributeError):
            wrapper.set("test")

    def test_wrapper_behavior_consistency_across_types(self):
        """Test that wrapper behavior is consistent across different value types."""
        test_values = ["string", 42, 3.14, True, None, [], {}, {"key": "value"}, [1, 2, 3]]

        for value in test_values:
            mock_status = Mock(spec=TaskStatus)
            wrapper = TaskStatusWrapper(mock_status)

            # First set should succeed
            result1 = wrapper.set(value)
            assert result1 is True
            assert wrapper.is_set is True

            # Second set should fail
            result2 = wrapper.set("different_value")
            assert result2 is False
            assert wrapper.is_set is True

            # Verify started was called with original value
            mock_status.started.assert_called_once_with(value)
