"""本文件同时有 mock 和 real_api，llm_client 按标记选密钥。"""
import os

import pytest


# ============ 请求参数传递：单轮 Prompt 转换 ============
@pytest.mark.mock
def test_mock_single_turn_prompt_to_messages(llm_client, mocker, request):
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
    result = llm_client.chat(prompt)

    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": result,
        "response_len": len(result),
        "expected_exception": None
    }

    assert result == reply
    mock_create.assert_called_once()
    assert mock_create.call_args.kwargs["messages"] == [
        {"role": "user", "content": prompt}
    ]


# ============ 请求参数传递：多轮对话 ============
@pytest.mark.mock
def test_mock_multi_turn_messages_passed(llm_client, mocker, request):
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
    result = llm_client.chat(messages=messages)

    request.node.stash["payload_data"] = {
        "multi_turn_messages": messages,
        "prompt": [messages[0]["content"], messages[-1]["content"]],
        "prompt_len": [len(messages[0]["content"]), len(messages[-1]["content"])],
        "response": result,
        "response_len": len(result),
        "expected_exception": None
    }

    assert result == reply
    mock_create.assert_called_once()
    called_messages = mock_create.call_args.kwargs["messages"]
    assert [item["role"] for item in called_messages] == ["user", "assistant", "user"]
    assert [item["content"] for item in called_messages] == [
        "我的名字是小明",
        "你好，小明",
        "我叫什么名字？",
    ]
    assert called_messages == messages


# ============ 请求参数传递：默认值 ============
@pytest.mark.mock
def test_mock_chat_default_request_params(llm_client, llm_config, mock_api_key, mocker, request):
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
    result = llm_client.chat(prompt)

    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": result,
        "response_len": len(result),
        "model": llm_config["model"],
        "base_url": llm_config["base_url"],
        "temperature": llm_config["temperature"],
        "max_tokens": llm_config["max_tokens"],
        "timeout": llm_config["timeout"],
        "expected_exception": None
    }

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


@pytest.mark.real_api
def test_real_api_uses_env_api_key(llm_client):
    """real_api 标记把 .env 中的真实 api_key 传给客户端，不发起对话"""
    env_api_key = os.getenv("DEEPSEEK_API_KEY")
    api_key_ok = bool(env_api_key) and str(llm_client.client.api_key) == env_api_key
    assert api_key_ok, "real_api 用例应使用 .env 的 DEEPSEEK_API_KEY"


# ============ 请求参数传递：调用时改写 ============
@pytest.mark.mock
def test_mock_chat_override_request_params(llm_client, mocker, request):
    """Mock用例：model、temperature、max_tokens、timeout 在 chat 调用时改写后按新值传递"""
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
    model = "mock-model"
    temperature = 0.2
    max_tokens = 128
    timeout = 8
    result = llm_client.chat(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": result,
        "response_len": len(result),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "expected_exception": None
    }

    assert result == reply
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == model
    assert kwargs["temperature"] == temperature
    assert kwargs["max_tokens"] == max_tokens
    assert kwargs["timeout"] == timeout
