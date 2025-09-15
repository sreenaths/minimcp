import json
from http import HTTPStatus

import pytest
from mcp import types

from minimcp.server.transports.http_transport_base import (
    CONTENT_TYPE_JSON,
    MCP_PROTOCOL_VERSION_HEADER,
    HTTPResult,
    HTTPTransportBase,
)


class TestHTTPTransportBase:
    """Test suite for HTTPTransportBase."""

    @pytest.fixture
    def transport_base(self):
        """Create an HTTPTransportBase instance."""
        return HTTPTransportBase()

    def test_http_result_dataclass(self):
        """Test HTTPResult dataclass."""
        # Test with minimal fields
        result = HTTPResult(HTTPStatus.OK)
        assert result.status_code == HTTPStatus.OK
        assert result.content is None
        assert result.media_type is None
        assert result.headers is None

        # Test with all fields
        result = HTTPResult(
            HTTPStatus.OK,
            content='{"result": "success"}',
            media_type="application/json",
            headers={"Custom-Header": "value"},
        )
        assert result.status_code == HTTPStatus.OK
        assert result.content == '{"result": "success"}'
        assert result.media_type == "application/json"
        assert result.headers == {"Custom-Header": "value"}

    def test_constants(self):
        """Test module constants."""
        assert MCP_PROTOCOL_VERSION_HEADER == "MCP-Protocol-Version"
        assert CONTENT_TYPE_JSON == "application/json"

    def test_handle_unsupported_request(self, transport_base):
        """Test _handle_unsupported_request method."""
        supported_methods = {"POST", "PUT"}

        result = transport_base._handle_unsupported_request(supported_methods)

        assert result.status_code == HTTPStatus.METHOD_NOT_ALLOWED
        assert result.headers is not None
        assert result.headers["Content-Type"] == CONTENT_TYPE_JSON
        assert "Allow" in result.headers
        # Should contain both methods
        allow_header = result.headers["Allow"]
        assert "POST" in allow_header
        assert "PUT" in allow_header

    def test_check_accept_headers_valid(self, transport_base):
        """Test _check_accept_headers with valid headers."""
        headers = {"Accept": "application/json, text/plain"}
        needed_types = {"application/json"}

        result = transport_base._check_accept_headers(headers, needed_types)
        assert result is None  # No error

    def test_check_accept_headers_invalid(self, transport_base):
        """Test _check_accept_headers with invalid headers."""
        headers = {"Accept": "text/plain, text/html"}
        needed_types = {"application/json"}

        result = transport_base._check_accept_headers(headers, needed_types)
        assert result is not None
        assert result.status_code == HTTPStatus.NOT_ACCEPTABLE
        assert "Not Acceptable" in result.content

    def test_check_accept_headers_missing(self, transport_base):
        """Test _check_accept_headers with missing Accept header."""
        headers = {}
        needed_types = {"application/json"}

        result = transport_base._check_accept_headers(headers, needed_types)
        assert result is not None
        assert result.status_code == HTTPStatus.NOT_ACCEPTABLE

    def test_check_accept_headers_with_quality_values(self, transport_base):
        """Test _check_accept_headers with quality values."""
        headers = {"Accept": "application/json; q=0.9, text/plain; q=0.1"}
        needed_types = {"application/json"}

        result = transport_base._check_accept_headers(headers, needed_types)
        assert result is None  # Should work - quality values are stripped

    def test_check_accept_headers_case_insensitive(self, transport_base):
        """Test _check_accept_headers is case insensitive."""
        headers = {"Accept": "APPLICATION/JSON"}
        needed_types = {"application/json"}

        result = transport_base._check_accept_headers(headers, needed_types)
        assert result is None

    def test_check_content_type_valid(self, transport_base):
        """Test _check_content_type with valid content type."""
        headers = {"Content-Type": "application/json"}

        result = transport_base._check_content_type(headers)
        assert result is None

    def test_check_content_type_invalid(self, transport_base):
        """Test _check_content_type with invalid content type."""
        headers = {"Content-Type": "text/plain"}

        result = transport_base._check_content_type(headers)
        assert result is not None
        assert result.status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
        assert "Unsupported Media Type" in result.content

    def test_check_content_type_missing(self, transport_base):
        """Test _check_content_type with missing Content-Type header."""
        headers = {}

        result = transport_base._check_content_type(headers)
        assert result is not None
        assert result.status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE

    def test_check_content_type_with_charset(self, transport_base):
        """Test _check_content_type with charset parameter."""
        headers = {"Content-Type": "application/json; charset=utf-8"}

        result = transport_base._check_content_type(headers)
        assert result is None

    def test_check_content_type_case_insensitive(self, transport_base):
        """Test _check_content_type is case insensitive."""
        headers = {"Content-Type": "APPLICATION/JSON"}

        result = transport_base._check_content_type(headers)
        assert result is None

    def test_validate_protocol_version_valid(self, transport_base):
        """Test _validate_protocol_version with valid version."""
        headers = {"MCP-Protocol-Version": "2025-06-18"}
        body = '{"jsonrpc": "2.0", "method": "test", "id": 1}'

        result = transport_base._validate_protocol_version(headers, body)
        assert result is None

    def test_validate_protocol_version_invalid(self, transport_base):
        """Test _validate_protocol_version with invalid version."""
        headers = {"MCP-Protocol-Version": "invalid-version"}
        body = '{"jsonrpc": "2.0", "method": "test", "id": 1}'

        result = transport_base._validate_protocol_version(headers, body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "Unsupported protocol version" in result.content

    def test_validate_protocol_version_missing_uses_default(self, transport_base):
        """Test _validate_protocol_version uses default when header is missing."""
        headers = {}
        body = '{"jsonrpc": "2.0", "method": "test", "id": 1}'

        result = transport_base._validate_protocol_version(headers, body)
        assert result is None  # Should use default version

    def test_validate_protocol_version_initialize_request_skipped(self, transport_base):
        """Test _validate_protocol_version skips validation for initialize requests."""
        headers = {"MCP-Protocol-Version": "invalid-version"}
        body = '{"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1}'

        result = transport_base._validate_protocol_version(headers, body)
        assert result is None  # Should skip validation for initialize

    def test_validate_protocol_version_malformed_json_ignored(self, transport_base):
        """Test _validate_protocol_version ignores malformed JSON."""
        headers = {"MCP-Protocol-Version": "2025-06-18"}
        body = "not valid json"

        result = transport_base._validate_protocol_version(headers, body)
        assert result is None  # Should ignore JSON parsing errors

    def test_validate_protocol_version_case_insensitive_header(self, transport_base):
        """Test _validate_protocol_version with case insensitive header."""
        headers = {"mcp-protocol-version": "2025-06-18"}
        body = '{"jsonrpc": "2.0", "method": "test", "id": 1}'

        result = transport_base._validate_protocol_version(headers, body)
        assert result is None

    def test_build_error_result(self, transport_base):
        """Test _build_error_result method."""
        status_code = HTTPStatus.BAD_REQUEST
        error_message = "Test error message"

        result = transport_base._build_error_result(status_code, types.PARSE_ERROR, error_message)

        assert result.status_code == status_code
        assert result.media_type == CONTENT_TYPE_JSON
        assert result.content is not None

        # Parse the error response
        error_response = json.loads(result.content)
        assert "error" in error_response
        # The error message includes the exception type prefix
        assert error_message in error_response["error"]["message"]

    def test_build_error_result_with_different_status_codes(self, transport_base):
        """Test _build_error_result with different HTTP status codes."""
        test_cases = [
            (HTTPStatus.BAD_REQUEST, "Bad request"),
            (HTTPStatus.NOT_FOUND, "Not found"),
            (HTTPStatus.INTERNAL_SERVER_ERROR, "Internal error"),
            (HTTPStatus.NOT_ACCEPTABLE, "Not acceptable"),
        ]

        for status_code, message in test_cases:
            result = transport_base._build_error_result(status_code, types.PARSE_ERROR, message)
            assert result.status_code == status_code
            assert message in result.content

    def test_multiple_needed_content_types(self, transport_base):
        """Test _check_accept_headers with multiple needed content types."""
        headers = {"Accept": "application/json, text/event-stream"}
        needed_types = {"application/json", "text/event-stream"}

        result = transport_base._check_accept_headers(headers, needed_types)
        assert result is None

    def test_partial_needed_content_types(self, transport_base):
        """Test _check_accept_headers when only some needed types are accepted."""
        headers = {"Accept": "application/json"}
        needed_types = {"application/json", "text/event-stream"}

        result = transport_base._check_accept_headers(headers, needed_types)
        assert result is not None
        assert result.status_code == HTTPStatus.NOT_ACCEPTABLE

    def test_empty_accept_header(self, transport_base):
        """Test _check_accept_headers with empty Accept header."""
        headers = {"Accept": ""}
        needed_types = {"application/json"}

        result = transport_base._check_accept_headers(headers, needed_types)
        assert result is not None
        assert result.status_code == HTTPStatus.NOT_ACCEPTABLE

    def test_whitespace_in_headers(self, transport_base):
        """Test header parsing with extra whitespace."""
        headers = {"Accept": " application/json , text/plain ", "Content-Type": " application/json ; charset=utf-8 "}

        # Accept header test
        result = transport_base._check_accept_headers(headers, {"application/json"})
        assert result is None

        # Content-Type header test
        result = transport_base._check_content_type(headers)
        assert result is None

    def test_validate_request_body_valid_json_rpc(self, transport_base):
        """Test _validate_request_body with valid JSON-RPC request."""
        body = '{"jsonrpc": "2.0", "method": "test", "id": 1}'

        result = transport_base._validate_request_body(body)
        assert result is None

    def test_validate_request_body_valid_json_rpc_with_params(self, transport_base):
        """Test _validate_request_body with valid JSON-RPC request including params."""
        body = '{"jsonrpc": "2.0", "method": "test", "params": {"key": "value"}, "id": 1}'

        result = transport_base._validate_request_body(body)
        assert result is None

    def test_validate_request_body_valid_notification(self, transport_base):
        """Test _validate_request_body with valid JSON-RPC notification (no id)."""
        body = '{"jsonrpc": "2.0", "method": "notification"}'

        result = transport_base._validate_request_body(body)
        assert result is None

    def test_validate_request_body_invalid_json(self, transport_base):
        """Test _validate_request_body with invalid JSON."""
        body = "not valid json"

        result = transport_base._validate_request_body(body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "Invalid JSON" in result.content

        # Parse the error response to check error code
        error_response = json.loads(result.content)
        assert error_response["error"]["code"] == types.PARSE_ERROR

    def test_validate_request_body_empty_json(self, transport_base):
        """Test _validate_request_body with empty JSON."""
        body = ""

        result = transport_base._validate_request_body(body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "Invalid JSON" in result.content

    def test_validate_request_body_json_array(self, transport_base):
        """Test _validate_request_body with JSON array instead of object."""
        body = '[{"jsonrpc": "2.0", "method": "test", "id": 1}]'

        result = transport_base._validate_request_body(body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "not a dictionary" in result.content

        # Parse the error response to check error code
        error_response = json.loads(result.content)
        assert error_response["error"]["code"] == types.PARSE_ERROR

    def test_validate_request_body_json_string(self, transport_base):
        """Test _validate_request_body with JSON string instead of object."""
        body = '"just a string"'

        result = transport_base._validate_request_body(body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "not a dictionary" in result.content

    def test_validate_request_body_json_number(self, transport_base):
        """Test _validate_request_body with JSON number instead of object."""
        body = "42"

        result = transport_base._validate_request_body(body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "not a dictionary" in result.content

    def test_validate_request_body_missing_jsonrpc(self, transport_base):
        """Test _validate_request_body with missing jsonrpc field."""
        body = '{"method": "test", "id": 1}'

        result = transport_base._validate_request_body(body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "Not a JSON-RPC message" in result.content

        # Parse the error response to check error code
        error_response = json.loads(result.content)
        assert error_response["error"]["code"] == types.INVALID_PARAMS

    def test_validate_request_body_wrong_jsonrpc_version(self, transport_base):
        """Test _validate_request_body with wrong JSON-RPC version."""
        body = '{"jsonrpc": "1.0", "method": "test", "id": 1}'

        result = transport_base._validate_request_body(body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "Not a JSON-RPC 2.0 message" in result.content

        # Parse the error response to check error code
        error_response = json.loads(result.content)
        assert error_response["error"]["code"] == types.INVALID_PARAMS

    def test_validate_request_body_null_jsonrpc(self, transport_base):
        """Test _validate_request_body with null jsonrpc field."""
        body = '{"jsonrpc": null, "method": "test", "id": 1}'

        result = transport_base._validate_request_body(body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "Not a JSON-RPC 2.0 message" in result.content

    def test_validate_request_body_empty_object(self, transport_base):
        """Test _validate_request_body with empty JSON object."""
        body = "{}"

        result = transport_base._validate_request_body(body)
        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "Not a JSON-RPC message" in result.content

    def test_validate_request_body_extra_fields(self, transport_base):
        """Test _validate_request_body with extra fields (should be valid)."""
        body = '{"jsonrpc": "2.0", "method": "test", "id": 1, "extra": "field"}'

        result = transport_base._validate_request_body(body)
        assert result is None  # Extra fields should be allowed

    def test_validate_request_body_unicode_content(self, transport_base):
        """Test _validate_request_body with unicode content."""
        body = '{"jsonrpc": "2.0", "method": "test", "params": {"message": "Hello 世界"}, "id": 1}'

        result = transport_base._validate_request_body(body)
        assert result is None

    def test_validate_request_body_nested_objects(self, transport_base):
        """Test _validate_request_body with nested objects in params."""
        body = '{"jsonrpc": "2.0", "method": "test", "params": {"nested": {"deep": {"value": 42}}}, "id": 1}'

        result = transport_base._validate_request_body(body)
        assert result is None

    def test_validate_request_body_malformed_json_cases(self, transport_base):
        """Test _validate_request_body with various malformed JSON cases."""
        malformed_cases = [
            '{"jsonrpc": "2.0", "method": "test", "id": 1',  # Missing closing brace
            '{"jsonrpc": "2.0", "method": "test", "id": 1,}',  # Trailing comma
            '{"jsonrpc": "2.0", "method": "test", "id": }',  # Missing value
            '{jsonrpc: "2.0", "method": "test", "id": 1}',  # Unquoted key
            '{"jsonrpc": "2.0", "method": \'test\', "id": 1}',  # Single quotes
        ]

        for malformed_body in malformed_cases:
            result = transport_base._validate_request_body(malformed_body)
            assert result is not None
            assert result.status_code == HTTPStatus.BAD_REQUEST
            assert "Invalid JSON" in result.content
