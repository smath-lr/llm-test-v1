import time
from email.utils import parsedate_to_datetime

import pytest
from httpx2 import ConnectError, ConnectTimeout, RemoteProtocolError
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

from src.llm_client import ATTEMPT_SECONDS, TOTAL_BUDGET_SECONDS, _TEMP_FAULT_CODES, _retry_interval


_JITTER = 0.5
_FUTURE_HTTP_DATE = "Tue, 15 Nov 2099 00:00:00 GMT"
_PAST_HTTP_DATE = "Sun, 06 Nov 1994 08:49:37 GMT"
_RETRY_OK = "重试后成功"


def _status_error(mocker, cls, status, body, headers=None, message="mock error"):
    """组装带状态码、body 和响应头的 OpenAI 状态异常。"""
    return cls(
        message=message,
        response=mocker.Mock(status_code=status, headers=headers or {}, request=None),
        body=body,
    )


def _connection_error(mocker, cause):
    """组装 APIConnectionError，并挂上指定 cause。cause 为 None 表示没有原因。"""
    exc = APIConnectionError(request=mocker.Mock())
    if cause is not None:
        exc.__cause__ = cause
    return exc


def _timeout_error(mocker):
    """组装 APITimeoutError。超时异常本身不重试。"""
    return APITimeoutError(request=mocker.Mock())


def _cause_by_name(name):
    """按名字造连接失败的 cause：连接拒绝、协议错误、连接超时、两种 DNS，或没有 cause。"""
    if name == "connect":
        return ConnectError("connection refused")
    if name == "remote":
        return RemoteProtocolError("peer closed")
    if name == "connect_timeout":
        return ConnectTimeout("connect timed out")
    if name == "dns":
        return ConnectError("nodename nor servname provided")
    if name == "dns_nss":
        return ConnectError("name or service not known")
    return None


class _StatusOnly(Exception):
    """只有状态码和响应头的异常，用来测 502、503、504，避免被认成 InternalServerError。"""

    def __init__(self, status, headers=None):
        self.status_code = status
        self.response = type("Resp", (), {"status_code": status, "headers": headers or {}})()


def _interval_exc(mocker, spec):
    """按用例规格造出交给 _retry_interval 或 chat() 的异常。"""
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
        return _timeout_error(mocker)
    raise AssertionError(kind)


# 不重试：间隔单测和 chat 共用这一张表。顺序与文档模块三一致。
_NO_RETRY_CASES = [
    ("TC-041", "400参数错误", {"kind": "status", "cls": BadRequestError, "status": 400, "body": {"code": 400, "message": "参数错误，触发400"}, "message": "Invalid request"}, BadRequestError),
    ("TC-042", "401认证失败", {"kind": "status", "cls": AuthenticationError, "status": 401, "body": {"code": 401, "message": "认证失败"}, "message": "Authentication error"}, AuthenticationError),
    ("TC-043", "403无权限", {"kind": "status", "cls": PermissionDeniedError, "status": 403, "body": {"code": 403, "message": "无权限"}, "message": "Permission denied"}, PermissionDeniedError),
    ("TC-044", "404资源不存在", {"kind": "status", "cls": NotFoundError, "status": 404, "body": {"code": 404, "message": "资源不存在"}, "message": "Not found"}, NotFoundError),
    ("TC-045", "429没有Retry-After", {"kind": "rate", "headers": {}, "body": {"code": 429, "message": "请求限流，触发429"}, "message": "Rate limit exceeded"}, RateLimitError),
    ("TC-046", "429空白Retry-After", {"kind": "rate", "headers": {"Retry-After": "  "}}, RateLimitError),
    ("TC-047", "429非数字Retry-After", {"kind": "rate", "headers": {"Retry-After": "abc"}}, RateLimitError),
    ("TC-048", "429负数Retry-After", {"kind": "rate", "headers": {"Retry-After": "-1"}}, RateLimitError),
    ("TC-049", "429过去的HTTP日期", {"kind": "rate", "headers": {"Retry-After": _PAST_HTTP_DATE}}, RateLimitError),
    ("TC-050", "429但不是RateLimitError", {"kind": "status", "cls": APIStatusError, "status": 429, "headers": {}, "message": "status 429"}, APIStatusError),
    ("TC-051", "500body不是字典", {"kind": "internal", "body": None, "assign_body": ["server_error"]}, InternalServerError),
    ("TC-052", "500的type和code未命中", {"kind": "internal", "body": {"type": "invalid_request_error", "code": "bad_request"}}, InternalServerError),
    ("TC-053", "500的code为整数", {"kind": "internal", "body": {"code": 500, "message": "服务内部异常"}, "message": "Server internal error"}, InternalServerError),
    ("TC-054", "500只在message", {"kind": "internal", "body": {"message": "server_error"}, "message": "server_error"}, InternalServerError),
    ("TC-055", "500但类型不符", {"kind": "status", "cls": APIStatusError, "status": 500, "body": {"type": "server_error"}, "message": "status 500"}, APIStatusError),
    ("TC-056", "APITimeoutError不重试", {"kind": "timeout"}, APITimeoutError),
    ("TC-057", "连接没有cause", {"kind": "conn", "cause": "none"}, APIConnectionError),
    ("TC-058", "连接cause为ConnectTimeout", {"kind": "conn", "cause": "connect_timeout"}, APIConnectionError),
    ("TC-059", "DNS nodename", {"kind": "conn", "cause": "dns"}, APIConnectionError),
    ("TC-060", "DNS name or service", {"kind": "conn", "cause": "dns_nss"}, APIConnectionError),
]


_RETRY_INTERVAL_CASES = [
    ("TC-061", {"kind": "rate", "headers": {"Retry-After": "1.5"}}, 1.5),
    ("TC-062", {"kind": "rate", "headers": {"Retry-After": _FUTURE_HTTP_DATE}}, "positive"),
    ("TC-063", {"kind": "internal", "body": {"type": "server_error"}}, "jitter"),
    ("TC-064", {"kind": "internal", "body": {"code": "INTERNAL_SERVER_ERROR"}}, "jitter"),
    ("TC-065", {"kind": "bare", "status": 502, "headers": {"Retry-After": "5"}}, "jitter"),
    ("TC-066", {"kind": "bare", "status": 504, "headers": {"Retry-After": "5"}}, "jitter"),
    ("TC-067", {"kind": "bare", "status": 503, "headers": {"Retry-After": "4"}}, 4.0),
    ("TC-068", {"kind": "bare", "status": 503, "headers": {}}, "jitter"),
    ("TC-069", {"kind": "conn", "cause": "connect"}, "jitter"),
    ("TC-070", {"kind": "conn", "cause": "remote"}, "jitter"),
]


_INTERVAL_CASES = _RETRY_INTERVAL_CASES + [
    (case_id, spec, None) for case_id, _title, spec, _exc_type in _NO_RETRY_CASES
]


def _assert_interval(result, expected):
    """核对间隔：jitter 表示固定抖动值，None 表示不重试，数字表示 Retry-After 秒数。"""
    if expected == "jitter":
        assert result == _JITTER
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
    frozen = time.time()
    mocker.patch("src.llm_client.time.time", return_value=frozen)
    mocker.patch("src.llm_client.random.uniform", return_value=_JITTER)
    result = _retry_interval(_interval_exc(mocker, spec), attempt_number=1)
    if expected == "positive":
        moment = parsedate_to_datetime(spec["headers"]["Retry-After"])
        expected = moment.timestamp() - frozen
        assert expected > 0
    _assert_interval(result, expected)


def _reply(mocker, text):
    """造一条正常的 chat 响应，content 为给定文本。"""
    return mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(content=text))])


def _patch_create(mocker):
    """拦截真正的 completions.create，让 chat() 不访问网络。"""
    return mocker.patch("openai.resources.chat.completions.Completions.create")


def _no_jitter(mocker):
    """把抖动固定为 0，重试用例不会真的等待。"""
    mocker.patch("src.llm_client._backoff_jitter", return_value=0)


_STATUS_BY_CASE = {
    "TC-041": 400,
    "TC-042": 401,
    "TC-043": 403,
    "TC-044": 404,
    "TC-045": 429,
    "TC-046": 429,
    "TC-047": 429,
    "TC-048": 429,
    "TC-049": 429,
    "TC-050": 429,
    "TC-051": 500,
    "TC-052": 500,
    "TC-053": 500,
    "TC-054": 500,
    "TC-055": 500,
}


def _watch_deadline(mocker):
    """记下客户端每次读取的单调时钟，用来事后计算剩余预算。"""
    real = time.monotonic
    calls = []

    def wrapped():
        value = real()
        calls.append(value)
        return value

    mocker.patch("src.llm_client.time.monotonic", side_effect=wrapped)
    return calls


def _remain(calls):
    """用第一次读到的时钟加上总预算，算出现在还剩多少秒。"""
    return calls[0] + TOTAL_BUDGET_SECONDS - time.monotonic()


def _assert_no_retry_details(case_id, exc):
    """按用例编号核对不重试的原因：状态码、Retry-After、body 或连接 cause。"""
    if case_id in _STATUS_BY_CASE:
        assert exc.status_code == _STATUS_BY_CASE[case_id]
    if case_id == "TC-046":
        raw = exc.response.headers.get("Retry-After")
        assert raw is not None and not str(raw).strip()
    elif case_id == "TC-047":
        with pytest.raises(ValueError):
            float(exc.response.headers.get("Retry-After"))
    elif case_id == "TC-048":
        assert float(exc.response.headers.get("Retry-After")) < 0
    elif case_id == "TC-049":
        assert exc.response.headers.get("Retry-After") == _PAST_HTTP_DATE
    elif case_id == "TC-051":
        assert not isinstance(exc.body, dict)
    elif case_id == "TC-052":
        body = exc.body
        assert isinstance(body, dict)
        for key in ("type", "code"):
            value = body.get(key)
            assert isinstance(value, str)
            assert value.strip().casefold() not in _TEMP_FAULT_CODES
    elif case_id == "TC-053":
        assert isinstance(exc.body.get("code"), int) and exc.body.get("code") == 500
    elif case_id == "TC-054":
        assert exc.body.get("message") == "server_error"
        assert "type" not in exc.body and "code" not in exc.body
    elif case_id == "TC-057":
        assert exc.__cause__ is None
    elif case_id == "TC-058":
        assert isinstance(exc.__cause__, ConnectTimeout)
    elif case_id == "TC-059":
        assert isinstance(exc.__cause__, ConnectError)
        assert "nodename nor servname provided" in str(exc.__cause__).lower()
    elif case_id == "TC-060":
        assert isinstance(exc.__cause__, ConnectError)
        assert "name or service not known" in str(exc.__cause__).lower()


def _fault_500(mocker):
    """造一个 body.type 为 server_error 的 500，这是会重试的临时故障。"""
    return _status_error(
        mocker,
        InternalServerError,
        500,
        {"type": "server_error"},
        message="temporary server error",
    )


def _rate_limit(mocker, retry_after):
    """造一个 429。retry_after 为 None 时不带头，否则写入 Retry-After。"""
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
    _assert_no_retry_details(case_id, exc)


# ============ 次数、预算、中途成功 ============
@pytest.mark.mock
def test_mock_chat_500_fault_fails_three_times(llm_client, mocker, record_payload):
    """TC-075：500 临时故障连续失败 3 次。第 3 次因次数停止，前两次预算仍够。"""
    clocks = _watch_deadline(mocker)
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
    assert _remain(clocks) > 0 + ATTEMPT_SECONDS


@pytest.mark.mock
def test_mock_chat_429_retry_after_exceeds_budget(llm_client, mocker, record_payload):
    """TC-071：Retry-After 为 30 秒，剩余预算不够，不重试。"""
    clocks = _watch_deadline(mocker)
    mock_create = _patch_create(mocker)
    exc = _rate_limit(mocker, "30")
    mock_create.side_effect = exc
    prompt = "429超过预算"
    record_payload(prompt, exc="RateLimitError")
    with pytest.raises(RateLimitError, match="Rate limit exceeded") as exc_info:
        llm_client.chat(prompt)
    assert exc_info.value is exc
    assert exc.response.headers.get("Retry-After") == "30"
    mock_create.assert_called_once()
    assert not (_remain(clocks) > 30 + ATTEMPT_SECONDS)


@pytest.mark.mock
@pytest.mark.parametrize("failures", [1, 2], ids=["失败1次后成功", "失败2次后成功"])
def test_mock_chat_500_fault_then_success(llm_client, mocker, record_payload, failures):
    """TC-076 / TC-077：500 临时故障中途成功。"""
    clocks = _watch_deadline(mocker)
    _no_jitter(mocker)
    mock_create = _patch_create(mocker)
    mock_create.side_effect = [_fault_500(mocker) for _ in range(failures)] + [_reply(mocker, _RETRY_OK)]
    prompt = "500临时故障后成功"
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)
    assert result == _RETRY_OK
    assert mock_create.call_count == failures + 1
    assert _remain(clocks) > 0 + ATTEMPT_SECONDS


@pytest.mark.mock
def test_mock_chat_429_second_retry_after_exceeds_budget(llm_client, mocker, record_payload):
    """TC-072：第一次 Retry-After 为 0，第二次为 24，不发起第三次。"""
    clocks = _watch_deadline(mocker)
    mock_create = _patch_create(mocker)
    first = _rate_limit(mocker, "0")
    second = _rate_limit(mocker, "24")
    mock_create.side_effect = [first, second]
    prompt = "429第二次超出预算"
    record_payload(prompt, exc="RateLimitError")
    with pytest.raises(RateLimitError, match="Rate limit exceeded") as exc_info:
        llm_client.chat(prompt)
    assert exc_info.value is second
    assert first.response.headers.get("Retry-After") == "0"
    assert second.response.headers.get("Retry-After") == "24"
    assert mock_create.call_count == 2
    assert not (_remain(clocks) > 24 + ATTEMPT_SECONDS)


@pytest.mark.mock
@pytest.mark.parametrize("failures", [1, 2], ids=["失败1次后成功", "失败2次后成功"])
def test_mock_chat_429_retry_after_zero_then_success(llm_client, mocker, record_payload, failures):
    """TC-073 / TC-074：429 的 Retry-After 为 0，中途成功。"""
    clocks = _watch_deadline(mocker)
    mock_create = _patch_create(mocker)
    failures_exc = [_rate_limit(mocker, "0") for _ in range(failures)]
    mock_create.side_effect = failures_exc + [_reply(mocker, _RETRY_OK)]
    prompt = "429短等待后成功"
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)
    assert result == _RETRY_OK
    assert mock_create.call_count == failures + 1
    for item in failures_exc:
        assert item.response.headers.get("Retry-After") == "0"
    assert _remain(clocks) > 0 + ATTEMPT_SECONDS
