import pytest
from openai import (
    BadRequestError,
    InternalServerError,
    RateLimitError,
)


# ============ 异常返回用例-不重试：API 429、400、500 ============
@pytest.mark.mock
def test_mock_chat_api_429_error(llm_client, mocker, record_payload):
    """Mock用例：模拟OpenAI服务端429限流，不重试，直接抛出 RateLimitError"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # APIStatusError 内部会取 response.request / response.headers，需一并补齐
    mock_create.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=mocker.Mock(status_code=429, headers={}, request=None),
        body={"code": 429, "message": "请求限流，触发429"},
    )
    prompt = "触发限流"
    record_payload(prompt, exc="RateLimitError")
    with pytest.raises(RateLimitError, match="Rate limit exceeded") as exc_info:
        llm_client.chat(prompt)

    assert exc_info.value.status_code == 429
    assert exc_info.value.body == {"code": 429, "message": "请求限流，触发429"}
    # 只调用1次，不重试
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_api_400_error(llm_client, mocker, record_payload):
    """Mock用例：模拟OpenAI服务端400参数错误，不重试，直接抛出 BadRequestError"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # APIStatusError 内部会取 response.request / response.headers，需一并补齐
    mock_create.side_effect = BadRequestError(
        message="Invalid request",
        response=mocker.Mock(status_code=400, headers={}, request=None),
        body={"code": 400, "message": "参数错误，触发400"},
    )

    prompt = "触发400"
    record_payload(prompt, exc="BadRequestError")
    with pytest.raises(BadRequestError, match="Invalid request") as exc_info:
        llm_client.chat(prompt)

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {"code": 400, "message": "参数错误，触发400"}
    # 只调用1次，不重试
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_server_500_error(llm_client, mocker, record_payload):
    """Mock用例：模拟OpenAI服务端500内部错误，不重试，直接抛出 InternalServerError"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # APIStatusError 内部会取 response.request / response.headers，需一并补齐
    mock_create.side_effect = InternalServerError(
        message="Server internal error",
        response=mocker.Mock(status_code=500, headers={}, request=None),
        body={"code": 500, "message": "服务内部异常"},
    )

    prompt = "触发服务端500"
    record_payload(prompt, exc="InternalServerError")

    with pytest.raises(InternalServerError, match="Server internal error") as exc_info:
        llm_client.chat(prompt)

    assert exc_info.value.status_code == 500
    assert exc_info.value.body == {"code": 500, "message": "服务内部异常"}
    # 只调用1次，不重试
    mock_create.assert_called_once()
