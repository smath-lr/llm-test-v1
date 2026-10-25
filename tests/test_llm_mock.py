import json
from openai import APIConnectionError, APIError, APITimeoutError
import pytest


@pytest.mark.mock
def test_mock_chat_normal_text(llm_client, mocker, valid_prompt, request):
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

    result = llm_client.chat(valid_prompt)
    request.node.stash["payload_data"] = {
        "prompt": valid_prompt,
        "prompt_len": len(valid_prompt),
        "response": result,
        "response_len": len(result),
        "expected_exception": None
    }

    assert result == "自动化测试可以提前发现代码逻辑问题"
    # 校验：确认底层create方法被调用了一次
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_return_json(llm_client, mocker, request):
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
    result = llm_client.chat(prompt)
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": result,
        "response_len": len(result),
        "expected_exception": None
    }
    resp_result = json.loads(result)
    assert resp_result["name"] == "莉莉"
    assert resp_result["skill"] == "测试与内观"
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_empty_content(llm_client, mocker, request):
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
    result = llm_client.chat(prompt)

    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": result,
        "response_len": len(result),
        "expected_exception": None
    }

    assert result == ""
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_whitespace_content(llm_client, mocker, request):
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
    result = llm_client.chat(prompt)

    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": result,
        "response_len": len(result),
        "expected_exception": None
    }

    assert result.isspace()
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_openai_api_error(llm_client, mocker, request):
    """Mock用例：模拟OpenAI服务端APIError（限流、密钥错误、服务不可用）
    重点：不让create返回正常对象，而是直接抛出异常
    """
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # 直接让patch的函数抛出APIError
    mock_create.side_effect = APIError(
        message="Rate limit exceeded",
        request=None,
        body={"code": 429, "message": "请求限流"}
    )

    prompt = "触发限流"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "请求限流"
    }
    with pytest.raises(APIError) as exc_info:
        llm_client.chat(prompt)

    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.body["code"] == 429
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_timeout_error(llm_client, mocker, request):
    """Mock用例：模拟调用LLM接口超时 APITimeoutError"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_create.side_effect = APITimeoutError(request=None)

    prompt = "长时间思考问题"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "APITimeoutError"
    }
    with pytest.raises(APITimeoutError):
        llm_client.chat(prompt)

    mock_create.assert_called()


@pytest.mark.mock
def test_mock_chat_server_500_error(llm_client, mocker, request):
    """Mock用例：模拟OpenAI服务端500内部错误"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_create.side_effect = APIError(
        message="Server internal error",
        request=None,
        body={"code": 500, "message": "服务内部异常"}
    )

    prompt = "触发服务端500"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "服务内部异常"
    }

    with pytest.raises(APIError, match="Server internal error"):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_prompt_empty(llm_client, mocker, request):
    """Mock用例：传入空prompt，本地参数校验拦截，不请求远端"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")

    prompt = ""
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "ValueError"
    }
    with pytest.raises(ValueError, match="prompt不能为空"):
        llm_client.chat(prompt)

    # 重点断言：校验函数根本没调用openai接口
    mock_create.assert_not_called()


@pytest.mark.mock
def test_mock_chat_prompt_too_long(llm_client, mocker, request):
    """Mock用例：超长prompt，本地长度校验拦截"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    prompt = "A" * 5000  # 假设上限设置小于5000
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": r"^prompt长度不能超过"
    }
    
    with pytest.raises(ValueError, match=r"^prompt长度不能超过"):
        llm_client.chat(prompt)
    mock_create.assert_not_called()


test_case_list = [
    ("简单问候", "你好呀", "你好呀"),
    ("数字返回", "输出数字123", "123"),
    ("英文返回", "输出hello", "hello world"),
]
@pytest.mark.mock
@pytest.mark.parametrize("case_title, prompt, expect_text", test_case_list)
def test_mock_chat_parametrize(llm_client, mocker, request, case_title, prompt, expect_text):
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

    result = llm_client.chat(prompt)

    request.node.stash["payload_data"] = {
        "case_title": case_title,
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": result,
        "response_len": len(result),
        "expected_exception": None
    }

    assert result == expect_text
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_response_missing_content(llm_client, mocker, request):
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
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "AttributeError"
    }

    # 预期：访问message.content时抛属性不存在异常
    with pytest.raises(AttributeError):
        llm_client.chat(prompt)

    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_retry_on_timeout(llm_client, mocker, request):
    """Mock用例：接口超时，触发重试，最多重试2次"""
    from openai import APITimeoutError
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # 连续抛出超时异常，触发重试
    mock_create.side_effect = APITimeoutError("request timeout")

    prompt = "测试超时"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "APITimeoutError"
    }
    with pytest.raises(APITimeoutError):
        llm_client.chat(prompt)
    # 原始调用1次 + 重试2次，合计调用3次
    mock_create.call_count == 3


@pytest.mark.mock
def test_mock_chat_retry_on_connection_error(llm_client, mocker, request):
    """Mock用例：连接失败，触发重试，最多重试2次"""
    from openai import APIConnectionError
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # 连续抛出连接失败异常，触发重试
    mock_create.side_effect = APIConnectionError(
        message="connection refuse",
        request=None
    )

    prompt = "测试连接异常"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "APIConnectionError"
    }
    with pytest.raises(APIConnectionError):
        llm_client.chat(prompt)
    # 原始调用1次 + 重试2次，合计调用3次
    mock_create.call_count == 3


@pytest.mark.mock
def test_mock_chat_no_retry_on_bad_request(llm_client, mocker, request):
    """Mock用例：400参数错误，不重试，直接抛出异常"""
    from openai import BadRequestError
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_create.side_effect = BadRequestError(
        message="invalid prompt param",
        response=mocker.Mock(status_code=400),
        body=None
    )

    prompt = "非法参数"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "BadRequestError"
    }
    with pytest.raises(BadRequestError, match="invalid prompt param"):
        llm_client.chat(prompt)
    # 只调用1次，没有重试
    mock_create.assert_called_once()

@pytest.mark.mock
def test_mock_chat_retry_success_after_timeout(llm_client, mocker, request):
    """Mock用例：前两次超时，第三次成功返回，重试后正常拿到结果"""
    from openai import APITimeoutError
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # 前两次超时和连接失败，第三次正常返回
    err1 = APITimeoutError(request=None)
    err1.custom_msg = "request timeout Error1"  #自定义标记
    err2 = APIConnectionError(request=None)
    err2.custom_msg = "connection refuse Error2"  #自定义标记

    mock_create.side_effect = [
        err1,err2,
        mocker.Mock(
            choices=[
                mocker.Mock(message=mocker.Mock(content="重试成功返回结果"))
            ]
        )
    ]

    prompt = "重试测试"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": "重试成功返回结果",
        "response_len": len("重试成功返回结果"),
        "expected_exception": None
    }
    resp = llm_client.chat(prompt)
    assert resp == "重试成功返回结果"
    # 一共调用3次：2次失败重试 + 最后一次成功
    mock_create.call_count == 3


# @pytest.mark.mock
# ===================== TODO：流式输出，当前chat暂未支持stream，待开发后解除注释 =====================
# def test_mock_chat_stream_response(llm_client, mocker, request):
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
#     result_stream = llm_client.chat(prompt, stream=True)
#     full_text = "".join([chunk.choices[0].delta.content for chunk in result_stream])
#
#     request.node.stash["payload_data"] = {
#         "prompt": prompt,
#         "prompt_len": len(prompt),
#         "response": full_text,
#         "response_len": len(full_text),
#         "expected_exception": None
#     }
#
#     assert full_text == "你好，这是流式测试"
#     mock_create.assert_called_once()
# ==============================================================================================
