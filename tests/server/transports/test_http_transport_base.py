import json
from http import HTTPStatus

import pytest

from minimcp.server.transports.http_transport_base import (
    CONTENT_TYPE_JSON,
    JSON_RPC_TO_HTTP_STATUS_CODES,
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

    def test_json_rpc_to_http_status_codes_mapping(self):
        """Test JSON-RPC error code to HTTP status code mapping."""
        import mcp.types as types

        assert JSON_RPC_TO_HTTP_STATUS_CODES[types.PARSE_ERROR] == HTTPStatus.BAD_REQUEST
        assert JSON_RPC_TO_HTTP_STATUS_CODES[types.INVALID_REQUEST] == HTTPStatus.BAD_REQUEST
        assert JSON_RPC_TO_HTTP_STATUS_CODES[types.INVALID_PARAMS] == HTTPStatus.BAD_REQUEST
        assert JSON_RPC_TO_HTTP_STATUS_CODES[types.METHOD_NOT_FOUND] == HTTPStatus.NOT_FOUND
        assert JSON_RPC_TO_HTTP_STATUS_CODES[types.INTERNAL_ERROR] == HTTPStatus.INTERNAL_SERVER_ERROR

    def test_constants(self):
        """Test module constants."""
        assert MCP_PROTOCOL_VERSION_HEADER == "MCP-Protocol-Version"
        assert CONTENT_TYPE_JSON == "application/json"

    def test_get_status_code_success_response(self, transport_base):
        """Test _get_status_code with successful response."""
        success_response = json.dumps({"jsonrpc": "2.0", "result": "success", "id": 1})

        status = transport_base._get_status_code(success_response)
        assert status == HTTPStatus.OK

    def test_get_status_code_error_responses(self, transport_base):
        """Test _get_status_code with various error responses."""
        # Parse error
        parse_error = json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None})
        status = transport_base._get_status_code(parse_error)
        assert status == HTTPStatus.BAD_REQUEST

        # Method not found
        method_not_found = json.dumps(
            {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": 1}
        )
        status = transport_base._get_status_code(method_not_found)
        assert status == HTTPStatus.NOT_FOUND

        # Internal error
        internal_error = json.dumps({"jsonrpc": "2.0", "error": {"code": -32603, "message": "Internal error"}, "id": 1})
        status = transport_base._get_status_code(internal_error)
        assert status == HTTPStatus.INTERNAL_SERVER_ERROR

    def test_get_status_code_unknown_error(self, transport_base):
        """Test _get_status_code with unknown error code."""
        unknown_error = json.dumps({"jsonrpc": "2.0", "error": {"code": -99999, "message": "Unknown error"}, "id": 1})

        status = transport_base._get_status_code(unknown_error)
        assert status == HTTPStatus.INTERNAL_SERVER_ERROR

    def test_get_status_code_malformed_json(self, transport_base):
        """Test _get_status_code with malformed JSON."""
        malformed_json = "not valid json"

        # Should handle JSON decode error gracefully and return 500
        status = transport_base._get_status_code(malformed_json)
        assert status == HTTPStatus.INTERNAL_SERVER_ERROR

    def test_get_status_code_non_dict_response(self, transport_base):
        """Test _get_status_code with non-dictionary JSON."""
        non_dict_response = json.dumps(["not", "a", "dict"])

        status = transport_base._get_status_code(non_dict_response)
        assert status == HTTPStatus.INTERNAL_SERVER_ERROR

    def test_get_status_code_error_without_code(self, transport_base):
        """Test _get_status_code with error object missing code."""
        error_without_code = json.dumps({"jsonrpc": "2.0", "error": {"message": "Error without code"}, "id": 1})

        status = transport_base._get_status_code(error_without_code)
        assert status == HTTPStatus.INTERNAL_SERVER_ERROR

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

        result = transport_base._build_error_result(status_code, error_message)

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
            result = transport_base._build_error_result(status_code, message)
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
