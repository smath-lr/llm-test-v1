import pytest
from httpx2 import ConnectError, ConnectTimeout, ReadTimeout, RemoteProtocolError
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from src.llm_client import _retry_interval


_JITTER = 0.5
_FUTURE_HTTP_DATE = "Tue, 15 Nov 2099 00:00:00 GMT"
_PAST_HTTP_DATE = "Sun, 06 Nov 1994 08:49:37 GMT"
_RETRY_OK = "重试后成功"


def _status_error(mocker, cls, status, body, headers=None, message="mock error"):
    return cls(
        message=message,
        response=mocker.Mock(status_code=status, headers=headers or {}, request=None),
        body=body,
    )


def _connection_error(mocker, cause):
    exc = APIConnectionError(request=mocker.Mock())
    if cause is not None:
        exc.__cause__ = cause
    return exc


def _timeout_error(mocker, cause):
    exc = APITimeoutError(request=mocker.Mock())
    if cause is not None:
        exc.__cause__ = cause
    return exc


def _cause_by_name(name):
    if name == "connect":
        return ConnectError("connection refused")
    if name == "remote":
        return RemoteProtocolError("peer closed")
    if name == "connect_timeout":
        return ConnectTimeout("connect timed out")
    if name == "read_timeout":
        return ReadTimeout("read timed out")
    if name == "dns":
        return ConnectError("nodename nor servname provided")
    if name == "dns_nss":
        return ConnectError("name or service not known")
    return None


class _StatusOnly(Exception):
    def __init__(self, status, headers=None):
        self.status_code = status
        self.response = type("Resp", (), {"status_code": status, "headers": headers or {}})()


def _interval_exc(mocker, spec):
    kind = spec["kind"]
    headers = spec.get("headers")
    if kind == "status":
        return _status_error(
            mocker,
            spec["cls"],
            spec["status"],
            spec.get("body"),
            headers,
            spec.get("message", "mock error"),
        )
    if kind == "rate":
        return _status_error(
            mocker,
            RateLimitError,
            429,
            spec.get("body"),
            headers,
            spec.get("message", "mock error"),
        )
    if kind == "internal":
        exc = _status_error(
            mocker,
            InternalServerError,
            500,
            spec.get("body"),
            headers,
            spec.get("message", "mock error"),
        )
        if "assign_body" in spec:
            exc.body = spec["assign_body"]
        return exc
    if kind == "bare":
        return _StatusOnly(spec["status"], headers)
    if kind == "conn":
        return _connection_error(mocker, _cause_by_name(spec["cause"]))
    if kind == "timeout":
        return _timeout_error(mocker, _cause_by_name(spec["cause"]))
    raise AssertionError(kind)


# 不重试：间隔单测和 chat 共用这一张表。顺序与文档模块三一致。
_NO_RETRY_CASES = [
    ("TC-041", "400参数错误", {"kind": "status", "cls": BadRequestError, "status": 400, "body": {"code": 400, "message": "参数错误，触发400"}, "message": "Invalid request"}, BadRequestError),
    ("TC-054", "401认证失败", {"kind": "status", "cls": AuthenticationError, "status": 401, "body": {"code": 401, "message": "认证失败"}, "message": "Authentication error"}, AuthenticationError),
    ("TC-055", "403无权限", {"kind": "status", "cls": PermissionDeniedError, "status": 403, "body": {"code": 403, "message": "无权限"}, "message": "Permission denied"}, PermissionDeniedError),
    ("TC-056", "404资源不存在", {"kind": "status", "cls": NotFoundError, "status": 404, "body": {"code": 404, "message": "资源不存在"}, "message": "Not found"}, NotFoundError),
    ("TC-042", "429没有Retry-After", {"kind": "rate", "headers": {}, "body": {"code": 429, "message": "请求限流，触发429"}, "message": "Rate limit exceeded"}, RateLimitError),
    ("TC-057", "429空白Retry-After", {"kind": "rate", "headers": {"Retry-After": "  "}}, RateLimitError),
    ("TC-058", "429非数字Retry-After", {"kind": "rate", "headers": {"Retry-After": "abc"}}, RateLimitError),
    ("TC-059", "429负数Retry-After", {"kind": "rate", "headers": {"Retry-After": "-1"}}, RateLimitError),
    ("TC-060", "429过去的HTTP日期", {"kind": "rate", "headers": {"Retry-After": _PAST_HTTP_DATE}}, RateLimitError),
    ("TC-061", "429但不是RateLimitError", {"kind": "status", "cls": APIStatusError, "status": 429, "headers": {}, "message": "status 429"}, APIStatusError),
    ("TC-062", "500body不是字典", {"kind": "internal", "body": None, "assign_body": ["server_error"]}, InternalServerError),
    ("TC-063", "500的type和code未命中", {"kind": "internal", "body": {"type": "invalid_request_error", "code": "bad_request"}}, InternalServerError),
    ("TC-043", "500的code为整数", {"kind": "internal", "body": {"code": 500, "message": "服务内部异常"}, "message": "Server internal error"}, InternalServerError),
    ("TC-044", "500只在message", {"kind": "internal", "body": {"message": "server_error"}, "message": "server_error"}, InternalServerError),
    ("TC-064", "500但类型不符", {"kind": "status", "cls": APIStatusError, "status": 500, "body": {"type": "server_error"}, "message": "status 500"}, APIStatusError),
    ("TC-065", "超时没有cause", {"kind": "timeout", "cause": "none"}, APITimeoutError),
    ("TC-047", "超时cause为ConnectTimeout", {"kind": "timeout", "cause": "connect_timeout"}, APITimeoutError),
    ("TC-045", "超时cause为ReadTimeout", {"kind": "timeout", "cause": "read_timeout"}, APITimeoutError),
    ("TC-066", "超时cause为ConnectError", {"kind": "timeout", "cause": "connect"}, APITimeoutError),
    ("TC-067", "超时cause为RemoteProtocolError", {"kind": "timeout", "cause": "remote"}, APITimeoutError),
    ("TC-068", "连接没有cause", {"kind": "conn", "cause": "none"}, APIConnectionError),
    ("TC-069", "连接cause为ConnectTimeout", {"kind": "conn", "cause": "connect_timeout"}, APIConnectionError),
    ("TC-046", "DNS nodename", {"kind": "conn", "cause": "dns"}, APIConnectionError),
    ("TC-070", "DNS name or service", {"kind": "conn", "cause": "dns_nss"}, APIConnectionError),
]


_RETRY_INTERVAL_CASES = [
    ("429秒数", {"kind": "rate", "headers": {"Retry-After": "1.5"}}, 1.5),
    ("429日期", {"kind": "rate", "headers": {"Retry-After": _FUTURE_HTTP_DATE}}, "positive"),
    ("500type命中", {"kind": "internal", "body": {"type": "server_error"}}, "jitter"),
    ("500code大小写命中", {"kind": "internal", "body": {"code": "INTERNAL_SERVER_ERROR"}}, "jitter"),
    ("502忽略Retry-After", {"kind": "bare", "status": 502, "headers": {"Retry-After": "5"}}, "jitter"),
    ("504忽略Retry-After", {"kind": "bare", "status": 504, "headers": {"Retry-After": "5"}}, "jitter"),
    ("503使用Retry-After", {"kind": "bare", "status": 503, "headers": {"Retry-After": "4"}}, 4.0),
    ("503没有Retry-After", {"kind": "bare", "status": 503, "headers": {}}, "jitter"),
    ("ConnectError", {"kind": "conn", "cause": "connect"}, "jitter"),
    ("RemoteProtocolError", {"kind": "conn", "cause": "remote"}, "jitter"),
]


_INTERVAL_CASES = _RETRY_INTERVAL_CASES + [
    (case_id, spec, None) for case_id, _title, spec, _exc_type in _NO_RETRY_CASES
]


def _assert_interval(result, expected):
    if expected == "jitter":
        assert result == _JITTER
        return
    if expected == "positive":
        assert isinstance(result, float) and result > 0
        return
    assert result == expected


@pytest.mark.mock
@pytest.mark.parametrize(
    "title, spec, expected",
    _INTERVAL_CASES,
    ids=[item[0] for item in _INTERVAL_CASES],
)
def test_retry_interval(mocker, title, spec, expected):
    """直接核对重试间隔：头里的秒数、抖动，或不可重试。"""
    mocker.patch("src.llm_client.random.uniform", return_value=_JITTER)
    result = _retry_interval(_interval_exc(mocker, spec), attempt_number=1)
    _assert_interval(result, expected)


def _reply(mocker, text):
    return mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(content=text))])


def _patch_create(mocker):
    return mocker.patch("openai.resources.chat.completions.Completions.create")


def _no_jitter(mocker):
    mocker.patch("src.llm_client._backoff_jitter", return_value=0)


def _assert_client_config(llm_client, llm_config):
    assert llm_client.model == llm_config["model"]
    assert llm_client.temperature == llm_config["temperature"]
    assert llm_client.max_tokens == llm_config["max_tokens"]
    assert llm_client.timeout == llm_config["timeout"]


def _fault_500(mocker):
    return _status_error(
        mocker,
        InternalServerError,
        500,
        {"type": "server_error"},
        message="temporary server error",
    )


def _rate_limit(mocker, retry_after):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return _status_error(
        mocker,
        RateLimitError,
        429,
        {"code": 429, "message": "请求限流，触发429"},
        headers=headers,
        message="Rate limit exceeded",
    )


# ============ 不重试 ============
@pytest.mark.mock
@pytest.mark.parametrize(
    "case_id, title, spec, exc_type",
    _NO_RETRY_CASES,
    ids=[item[0] for item in _NO_RETRY_CASES],
)
def test_mock_chat_no_retry(llm_client, mocker, record_payload, case_id, title, spec, exc_type):
    """不重试：抛出原异常，create 只调用 1 次。"""
    mock_create = _patch_create(mocker)
    exc = _interval_exc(mocker, spec)
    mock_create.side_effect = exc
    prompt = f"{case_id} {title}"
    record_payload(prompt, exc=exc_type.__name__)
    with pytest.raises(exc_type) as exc_info:
        llm_client.chat(prompt)
    assert exc_info.value is exc
    mock_create.assert_called_once()


# ============ 次数、预算、中途成功 ============
@pytest.mark.mock
def test_mock_chat_500_fault_fails_three_times(llm_client, llm_config, mocker, record_payload):
    """TC-048：500 临时故障连续失败 3 次。"""
    _no_jitter(mocker)
    mock_create = _patch_create(mocker)
    mock_create.side_effect = [_fault_500(mocker) for _ in range(3)]
    prompt = "500临时故障耗尽"
    record_payload(prompt, exc="InternalServerError")
    with pytest.raises(InternalServerError, match="temporary server error") as exc_info:
        llm_client.chat(prompt)
    assert exc_info.value.status_code == 500
    assert exc_info.value.body == {"type": "server_error"}
    assert mock_create.call_count == 3
    _assert_client_config(llm_client, llm_config)


@pytest.mark.mock
def test_mock_chat_429_retry_after_exceeds_budget(llm_client, mocker, record_payload):
    """TC-049：Retry-After 为 30 秒，预算不够，不重试。"""
    mock_create = _patch_create(mocker)
    mock_create.side_effect = _rate_limit(mocker, "30")
    prompt = "429超过预算"
    record_payload(prompt, exc="RateLimitError")
    with pytest.raises(RateLimitError, match="Rate limit exceeded"):
        llm_client.chat(prompt)
    mock_create.assert_called_once()


@pytest.mark.mock
@pytest.mark.parametrize("failures", [1, 2], ids=["失败1次后成功", "失败2次后成功"])
def test_mock_chat_500_fault_then_success(llm_client, llm_config, mocker, record_payload, failures):
    """TC-050 / TC-051：500 临时故障中途成功。"""
    _no_jitter(mocker)
    mock_create = _patch_create(mocker)
    mock_create.side_effect = [_fault_500(mocker) for _ in range(failures)] + [_reply(mocker, _RETRY_OK)]
    prompt = "500临时故障后成功"
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)
    assert result == _RETRY_OK
    assert mock_create.call_count == failures + 1
    if failures == 1:
        _assert_client_config(llm_client, llm_config)


@pytest.mark.mock
@pytest.mark.parametrize("failures", [1, 2], ids=["失败1次后成功", "失败2次后成功"])
def test_mock_chat_429_retry_after_zero_then_success(llm_client, mocker, record_payload, failures):
    """TC-052 / TC-053：429 的 Retry-After 为 0，中途成功。"""
    mock_create = _patch_create(mocker)
    mock_create.side_effect = [_rate_limit(mocker, "0") for _ in range(failures)] + [_reply(mocker, _RETRY_OK)]
    prompt = "429短等待后成功"
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)
    assert result == _RETRY_OK
    assert mock_create.call_count == failures + 1
