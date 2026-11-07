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
def test_mock_chat_return_json(llm_client, mocker, record_payload):
    """Mock用例1：模拟LLM返回json字符串，测试json解析链路"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")

    mock_response = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(
                    content='{"name":"莉莉","skill":"测试与内观"}'
                )
            )
        ]
        )
    mock_create.return_value = mock_response

    prompt = "请输出JSON格式信息"
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)
    resp_result = json.loads(result)
    assert resp_result["name"] == "莉莉"
    assert resp_result["skill"] == "测试与内观"
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_empty_content(llm_client, mocker, record_payload):
    """Mock用例：模拟LLM返回空字符串，校验上层chat函数处理空返回"""
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
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)

    assert result == ""
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_whitespace_content(llm_client, mocker, record_payload):
    """Mock用例：模拟LLM只返回空格、换行这类空白字符"""
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
    record_payload(prompt)
    result = llm_client.chat(prompt)
    record_payload(prompt, result)

    assert result.isspace()
    mock_create.assert_called_once()


test_case_list = [
    ("简单问候", "你好呀", "你好呀"),
    ("数字返回", "输出数字123", "123"),
    ("英文返回", "输出hello", "hello world"),
]
@pytest.mark.mock
@pytest.mark.parametrize("case_title, prompt, expect_text", test_case_list)
def test_mock_chat_parametrize(llm_client, mocker, record_payload, case_title, prompt, expect_text):
    """Mock综合用例：参数化批量模拟多种正常返回，减少重复样板代码"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_response = mocker.Mock(
        choices=[
            mocker.Mock(
                message=mocker.Mock(content=expect_text)
            )
        ]
    )
    mock_create.return_value = mock_response

    record_payload(prompt, case_title=case_title)
    result = llm_client.chat(prompt)
    record_payload(prompt, result, case_title=case_title)

    assert result == expect_text
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


# @pytest.mark.mock
# ===================== TODO：流式输出，当前chat暂未支持stream，待开发后解除注释 =====================
# def test_mock_chat_stream_response(llm_client, mocker, record_payload):
#     """Mock用例：模拟流式输出，返回多个chunk片段
#     前置依赖：改造 utils.llm_client.chat，增加 stream 参数，支持迭代返回chunk
#     """
#     mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
#
#     # 构造流式迭代器，每一块是一个chunk对象
#     chunk_1 = mocker.Mock(choices=[mocker.Mock(delta=mocker.Mock(content="你好"))])
#     chunk_2 = mocker.Mock(choices=[mocker.Mock(delta=mocker.Mock(content="，这是流式"))])
#     chunk_3 = mocker.Mock(choices=[mocker.Mock(delta=mocker.Mock(content="测试"))])
#
#     mock_create.return_value = iter([chunk_1, chunk_2, chunk_3])
#
#     prompt = "流式输出"
#     record_payload(prompt)
#     result_stream = llm_client.chat(prompt, stream=True)
#     full_text = "".join([chunk.choices[0].delta.content for chunk in result_stream])
#     record_payload(prompt, full_text)
#
#     assert full_text == "你好，这是流式测试"
#     mock_create.assert_called_once()
# ==============================================================================================
