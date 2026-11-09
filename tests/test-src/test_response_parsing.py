import json

import pytest


# ============ 服务端响应解析 ============
@pytest.mark.mock
def test_mock_chat_normal_text(llm_client, mocker, valid_prompt, record_payload):
    """Mock用例1：模拟LLM返回普通文本，验证chat正常拿到结果"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")

    mock_response = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(content="自动化测试可以提前发现代码逻辑问题")
            )
        ]
    )
    mock_create.return_value = mock_response

    record_payload(valid_prompt)
    result = llm_client.chat(valid_prompt)
    record_payload(valid_prompt, result)

    assert result == "自动化测试可以提前发现代码逻辑问题"
    # 校验：确认底层create方法被调用了一次
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_empty_content(llm_client, mocker, record_payload):
    """Mock用例：模拟LLM返回空字符串，校验chat拒绝并只请求一次"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_response = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(content="")
            )
        ]
    )
    mock_create.return_value = mock_response

    prompt = "返回空结果"
    record_payload(prompt, exc="RuntimeError")
    with pytest.raises(RuntimeError, match="content必须为去空格后非空的字符串"):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_whitespace_content(llm_client, mocker, record_payload):
    """Mock用例：模拟LLM只返回空格、换行这类空白字符，校验chat拒绝并只请求一次"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_response = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(content="   \n  \t  ")
            )
        ]
    )
    mock_create.return_value = mock_response

    prompt = "返回空白"
    record_payload(prompt, exc="RuntimeError")
    with pytest.raises(RuntimeError, match="content必须为去空格后非空的字符串"):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
@pytest.mark.parametrize("content", [1, None], ids=["整数", "None"])
def test_mock_chat_content_not_string(llm_client, mocker, record_payload, content):
    """Mock用例：content 不是字符串，校验 chat 拒绝并只请求一次"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_response = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(content=content)
            )
        ]
    )
    mock_create.return_value = mock_response

    prompt = "content不是字符串"
    record_payload(prompt, exc="RuntimeError")
    with pytest.raises(RuntimeError, match="content必须为去空格后非空的字符串"):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_response_empty_choices(llm_client, mocker, valid_prompt, record_payload):
    """Mock用例：返回空choices，抛出 RuntimeError 且不重试"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_response = mocker.Mock(choices=[])
    mock_create.return_value = mock_response

    prompt = "返回空choices"
    record_payload(prompt, exc="RuntimeError")
    with pytest.raises(RuntimeError, match="API 返回空choices"):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_response_missing_choices(llm_client, mocker, record_payload):
    """Mock用例：返回结果没有 choices 属性，抛出 RuntimeError 且不重试"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_create.return_value = mocker.Mock(spec=[])

    prompt = "返回结果没有choices"
    record_payload(prompt, exc="RuntimeError")
    with pytest.raises(RuntimeError, match="返回结果没有choices属性"):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_response_choices_not_list(llm_client, mocker, record_payload):
    """Mock用例：choices 能取到但不是列表，抛出 RuntimeError 且不重试"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_response = mocker.Mock(choices={"content": "你好"})
    mock_create.return_value = mock_response

    prompt = "choices不是列表"
    record_payload(prompt, exc="RuntimeError")
    with pytest.raises(RuntimeError, match="choices必须为列表"):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_response_missing_message(llm_client, mocker, record_payload):
    """Mock用例：第一条 choice 取不到 message，抛出 RuntimeError 且不重试"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_response = mocker.Mock(
        choices=[
            mocker.Mock(spec=[])
        ]
    )
    mock_create.return_value = mock_response

    prompt = "返回缺少message"
    record_payload(prompt, exc="RuntimeError")
    with pytest.raises(RuntimeError, match="响应缺少message"):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_response_missing_content(llm_client, mocker, record_payload):
    """Mock用例：message 上取不到 content，抛出 RuntimeError 且不重试"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # 重点：message这个mock对象，不设置content
    mock_response = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(spec=[])
            )
        ]
    )
    mock_create.return_value = mock_response

    prompt = "返回缺少content"
    record_payload(prompt, exc="RuntimeError")

    with pytest.raises(RuntimeError, match="响应缺少content"):
        llm_client.chat(prompt)

    mock_create.assert_called_once()

