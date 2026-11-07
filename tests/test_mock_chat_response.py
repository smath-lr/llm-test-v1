import pytest


# ============ 异常返回用例-不重试：返回结构残缺 ============
@pytest.mark.mock
def test_mock_chat_response_missing_content(llm_client, mocker, record_payload):
    """Mock用例：返回结构残缺，message对象不存在content属性，校验chat容错逻辑"""
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

    prompt = "返回残缺结构"
    record_payload(prompt, exc="AttributeError")

    # 预期：访问message.content时抛属性不存在异常
    with pytest.raises(AttributeError):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_response_empty_choices(llm_client, mocker, valid_prompt, record_payload):
    """Mock用例：返回空choices，校验chat容错逻辑"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_response = mocker.Mock(choices=[])
    mock_create.return_value = mock_response

    prompt = "返回空choices"
    record_payload(prompt, exc="RuntimeError")
    with pytest.raises(RuntimeError):
        llm_client.chat(prompt)

    mock_create.assert_called_once()