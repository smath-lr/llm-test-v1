import json
from openai import (
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
import pytest

from utils.llm_client import MAX_PROMPT_LENGTH


# ============ 正常返回用例 ============
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


# ============ 异常返回用例-不重试：prompt 参数校验 ============
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
@pytest.mark.parametrize("prompt, messages, match", [
    (None, None, "prompt和messages必须传一个"),
    ("你好", [{"role": "user", "content": "你好"}], "prompt和messages只能传一个"),
    (1, None, "prompt必须为字符串"),
    ("", None, "prompt不能为空"),
    ("  ", None, "prompt不能为空"),
    (None, {}, "messages必须为消息列表"),
    (None, [], "messages不能为空列表"),
    (None, ["你好"], "第1条message格式不正确"),
    (None, [{"content": "你好"}], "第1条message缺少role"),
    (None, [{"role": None, "content": "你好"}], "第1条message缺少role"),
    (None, [{"role": "user"}], "第1条message缺少content"),
    (None, [{"role": "user", "content": None}], "第1条message缺少content"),
    (None, [{"role": "user", "content": "你好", "name": "小明"}], "第1条message只能包含role和content"),
    (None, [{"role": "assistant", "content": "你好"}], "第1条message的role必须为user"),
    (None, [{"role": "system", "content": "你好"}], "第1条message的role必须为user"),
    (
        None,
        [
            {"role": "user", "content": "你好"},
            {"role": "user", "content": "在吗"},
            {"role": "user", "content": "继续"},
        ],
        "第2条message的role必须为assistant",
    ),
    (
        None,
        [
            {"role": "user", "content": "你好"},
            {"role": "system", "content": "在吗"},
            {"role": "user", "content": "继续"},
        ],
        "第2条message的role必须为assistant",
    ),
    (None, [{"role": "user", "content": ""}], "prompt不能为空"),
    (None, [{"role": "user", "content": "  "}], "prompt不能为空"),
    (None, [{"role": "user", "content": 1}], "第1条message的content必须为字符串"),
    (
        None,
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": 1},
            {"role": "user", "content": "继续"},
        ],
        "第2条message的content必须为字符串",
    ),
    (
        None,
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": ["在"]},
            {"role": "user", "content": "继续"},
        ],
        "第2条message的content必须为字符串",
    ),
])
def test_mock_chat_prompt_structure_invalid(llm_client, mocker, request, prompt, messages, match):
    """Mock用例：prompt 或 messages 结构不合法时本地拦截，不请求远端"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    shown = messages if messages is not None else prompt
    request.node.stash["payload_data"] = {
        "prompt": shown,
        "prompt_len": len(shown) if isinstance(shown, (str, list)) else None,
        "response": None,
        "response_len": 0,
        "expected_exception": "ValueError",
    }
    with pytest.raises(ValueError, match=match):
        llm_client.chat(prompt=prompt, messages=messages)
    mock_create.assert_not_called()


@pytest.mark.mock
def test_mock_chat_messages_must_be_odd_count(llm_client, mocker, request):
    """Mock用例：检测 messages 条数。合法条数是 1、3、5……这种单数。

    用例数据：2 条，user 后接 assistant，角色位置正确，但条数是偶数。
    提示：messages条数必须为1、3、5……这种单数。
    检测点：按条数拦截，不请求远端。
    """
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "在吗"},
    ]
    request.node.stash["payload_data"] = {
        "prompt": messages,
        "prompt_len": len(messages),
        "response": None,
        "response_len": 0,
        "expected_exception": "ValueError",
    }
    with pytest.raises(ValueError, match="messages条数必须为1、3、5……这种单数"):
        llm_client.chat(messages=messages)
    mock_create.assert_not_called()


def _user_messages(length):
    return [{"role": "user", "content": "A" * length}]


@pytest.mark.mock
@pytest.mark.parametrize("case_title, prompt, messages, length", [
    ("字符串长度1", "A" * 1, None, 1),
    ("字符串长度2", "A" * 2, None, 2),
    ("字符串长度199", "A" * (MAX_PROMPT_LENGTH - 1), None, MAX_PROMPT_LENGTH - 1),
    ("字符串长度200", "A" * MAX_PROMPT_LENGTH, None, MAX_PROMPT_LENGTH),
    ("字符串长度201", "A" * (MAX_PROMPT_LENGTH + 1), None, MAX_PROMPT_LENGTH + 1),
    ("一条messages的user内容长度201", None, _user_messages(MAX_PROMPT_LENGTH + 1), MAX_PROMPT_LENGTH + 1),
    (
        "三条messages的user内容长度201",
        None,
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "收到"},
            {"role": "user", "content": "A" * (MAX_PROMPT_LENGTH + 1)},
        ],
        MAX_PROMPT_LENGTH + 1,
    ),
])
def test_mock_chat_prompt_length(llm_client, mocker, request, case_title, prompt, messages, length):
    """Mock用例：字符串 prompt 覆盖长度 1、2、199、200、201；messages 只覆盖 user 内容长度 201"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    reply = "长度校验通过"
    mock_create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(content=reply))]
    )
    shown = messages if messages is not None else prompt
    too_long = length > MAX_PROMPT_LENGTH
    request.node.stash["payload_data"] = {
        "case_title": case_title,
        "prompt": shown,
        "prompt_len": length,
        "response": None if too_long else reply,
        "response_len": 0 if too_long else len(reply),
        "expected_exception": r"^prompt长度不能超过" if too_long else None,
    }
    if too_long:
        with pytest.raises(ValueError, match=r"^prompt长度不能超过"):
            llm_client.chat(prompt=prompt, messages=messages)
        mock_create.assert_not_called()
        return
    result = llm_client.chat(prompt=prompt, messages=messages)
    assert result == reply
    mock_create.assert_called_once()


# ============ 异常返回用例-不重试：返回结构残缺 ============
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


# ============ 异常返回用例-不重试：API 429、400、500 ============
@pytest.mark.mock
def test_mock_chat_api_429_error(llm_client, mocker, request):
    """Mock用例：模拟OpenAI服务端429限流，不重试，直接抛出 RateLimitError"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # APIStatusError 内部会取 response.request / response.headers，需一并补齐
    mock_create.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=mocker.Mock(status_code=429, headers={}, request=None),
        body={"code": 429, "message": "请求限流，触发429"},
    )
    prompt = "触发限流"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "RateLimitError",
    }
    with pytest.raises(RateLimitError, match="Rate limit exceeded") as exc_info:
        llm_client.chat(prompt)

    assert exc_info.value.status_code == 429
    assert exc_info.value.body == {"code": 429, "message": "请求限流，触发429"}
    # 只调用1次，不重试
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_api_400_error(llm_client, mocker, request):
    """Mock用例：模拟OpenAI服务端400参数错误，不重试，直接抛出 BadRequestError"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # APIStatusError 内部会取 response.request / response.headers，需一并补齐
    mock_create.side_effect = BadRequestError(
        message="Invalid request",
        response=mocker.Mock(status_code=400, headers={}, request=None),
        body={"code": 400, "message": "参数错误，触发400"},
    )

    prompt = "触发400"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "BadRequestError",
    }
    with pytest.raises(BadRequestError, match="Invalid request") as exc_info:
        llm_client.chat(prompt)

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {"code": 400, "message": "参数错误，触发400"}
    # 只调用1次，不重试
    mock_create.assert_called_once()


@pytest.mark.mock
def test_mock_chat_server_500_error(llm_client, mocker, request):
    """Mock用例：模拟OpenAI服务端500内部错误，不重试，直接抛出 InternalServerError"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    # APIStatusError 内部会取 response.request / response.headers，需一并补齐
    mock_create.side_effect = InternalServerError(
        message="Server internal error",
        response=mocker.Mock(status_code=500, headers={}, request=None),
        body={"code": 500, "message": "服务内部异常"},
    )

    prompt = "触发服务端500"
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response": None,
        "response_len": 0,
        "expected_exception": "InternalServerError",
    }

    with pytest.raises(InternalServerError, match="Server internal error") as exc_info:
        llm_client.chat(prompt)

    assert exc_info.value.status_code == 500
    assert exc_info.value.body == {"code": 500, "message": "服务内部异常"}
    # 只调用1次，不重试
    mock_create.assert_called_once()


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
