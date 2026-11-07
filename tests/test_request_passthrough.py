import os

import pytest


# ============ 单轮 / 多轮请求组装与参数透传 ============
# ============ 单轮 Prompt 转换 ============
@pytest.mark.mock
def test_mock_single_turn_prompt_to_messages(llm_client, mocker, record_payload):
    """Mock用例：单轮 prompt 转换为 messages，role 为 user，content 为原 prompt"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    reply = "单轮参数传递成功"
    mock_create.return_value = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(content=reply)
            )
        ]
    )

    prompt = "简单介绍自动化测试"
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)

    assert result == reply
    mock_create.assert_called_once()
    called_messages = mock_create.call_args.kwargs["messages"]
    assert isinstance(called_messages, list)
    assert len(called_messages) == 1
    assert isinstance(called_messages[0], dict)
    assert set(called_messages[0]) == {"role", "content"}
    assert called_messages[0]["role"] == "user"
    assert called_messages[0]["content"] == prompt


# ============ 请求参数传递：多轮对话 ============
@pytest.mark.mock
def test_mock_multi_turn_messages_passed(llm_client, mocker, record_payload):
    """Mock用例：多轮 messages 的角色、顺序、内容原样传给 create"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    reply = "多轮参数传递成功"
    mock_create.return_value = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(content=reply)
            )
        ]
    )

    messages = [
        {"role": "user", "content": "我的名字是小明"},
        {"role": "assistant", "content": "你好，小明"},
        {"role": "user", "content": "我叫什么名字？"},
    ]
    expected_messages = [dict(item) for item in messages]
    record_payload(messages)
    result = llm_client.chat(messages=messages)
    record_payload(messages, result)

    assert result == reply
    mock_create.assert_called_once()
    called_messages = mock_create.call_args.kwargs["messages"]
    assert isinstance(called_messages, list)
    assert len(called_messages) == len(expected_messages)
    for index, item in enumerate(called_messages):
        assert isinstance(item, dict)
        assert set(item) == {"role", "content"}
        assert item["role"] == expected_messages[index]["role"]
        assert item["content"] == expected_messages[index]["content"]


# ============ 请求参数传递：默认值 ============
@pytest.mark.mock
def test_mock_chat_default_request_params(llm_client, llm_config, mock_api_key, mocker, record_payload):
    """Mock用例：未改写时 model、timeout、temperature、max_tokens、base_url 走配置；mock 标记使用假 api_key"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    reply = "默认参数传递成功"
    mock_create.return_value = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(content=reply)
            )
        ]
    )

    prompt = "校验默认请求参数"
    sent = {
        "model": llm_config["model"],
        "base_url": llm_config["base_url"],
        "temperature": llm_config["temperature"],
        "max_tokens": llm_config["max_tokens"],
        "timeout": llm_config["timeout"],
    }
    record_payload(prompt, **sent)
    result = llm_client.chat(prompt)
    record_payload(prompt, result, **sent)

    assert result == reply
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == llm_config["model"]
    assert kwargs["temperature"] == llm_config["temperature"]
    assert kwargs["max_tokens"] == llm_config["max_tokens"]
    assert kwargs["timeout"] == llm_config["timeout"]
    assert str(llm_client.client.base_url) == str(llm_config["base_url"])
    # 只断言是否一致，避免失败报文打出真实 api_key
    env_api_key = os.getenv("DEEPSEEK_API_KEY")
    api_key_ok = str(llm_client.client.api_key) == mock_api_key and str(llm_client.client.api_key) != env_api_key
    assert api_key_ok, "mock 用例应使用假 api_key"


# ============ 请求参数传递：调用时改写 ============
_REQUEST_FIELDS = ("model", "temperature", "max_tokens", "timeout")
_OVERRIDE_CASES = [
    ("只覆盖model", {"model": "mock-model"}),
    ("只覆盖temperature", {"temperature": 0.2}),
    ("只覆盖max_tokens", {"max_tokens": 128}),
    ("只覆盖timeout", {"timeout": 8}),
    ("覆盖model和temperature", {"model": "mock-model", "temperature": 0.2}),
    ("覆盖全部请求参数", {
        "model": "mock-model",
        "temperature": 0.2,
        "max_tokens": 128,
        "timeout": 8,
    }),
]


@pytest.mark.mock
@pytest.mark.parametrize("case_title, overrides", _OVERRIDE_CASES)
def test_mock_chat_override_request_params(llm_client, llm_config, mocker, record_payload, case_title, overrides):
    """Mock用例：传入的参数覆盖默认值，未传入的参数仍使用配置默认值"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    reply = "改写参数传递成功"
    mock_create.return_value = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(content=reply)
            )
        ]
    )

    prompt = "校验改写后的请求参数"
    record_payload(prompt, case_title=case_title, overrides=overrides)
    result = llm_client.chat(prompt, **overrides)
    record_payload(prompt, result, case_title=case_title, overrides=overrides)

    assert result == reply
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    for field in _REQUEST_FIELDS:
        if field in overrides:
            assert kwargs[field] == overrides[field]
            assert kwargs[field] != llm_config[field]
        else:
            assert kwargs[field] == llm_config[field]
    assert llm_client.model == llm_config["model"]
    assert llm_client.temperature == llm_config["temperature"]
    assert llm_client.max_tokens == llm_config["max_tokens"]
    assert llm_client.timeout == llm_config["timeout"]
