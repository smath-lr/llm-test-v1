import pytest

from src.llm_client import MAX_PROMPT_LENGTH


# ============ Prompt / messages 输入校验 ============
@pytest.mark.mock
def test_mock_chat_prompt_empty(llm_client, mocker, record_payload):
    """Mock用例：传入空prompt，本地参数校验拦截，不请求远端"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")

    prompt = ""
    record_payload(prompt, exc="ValueError")
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
def test_mock_chat_prompt_structure_invalid(llm_client, mocker, record_payload, prompt, messages, match):
    """Mock用例：prompt 或 messages 结构不合法时本地拦截，不请求远端"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    record_payload(
        prompt,
        exc="ValueError",
        messages=messages,
    )
    with pytest.raises(ValueError, match=match):
        llm_client.chat(prompt=prompt, messages=messages)
    mock_create.assert_not_called()


@pytest.mark.mock
def test_mock_chat_messages_must_be_odd_count(llm_client, mocker, record_payload):
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
    record_payload(messages=messages, exc="ValueError")
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
def test_mock_chat_prompt_length(llm_client, mocker, record_payload, case_title, prompt, messages, length):
    """Mock用例：字符串 prompt 覆盖长度 1、2、199、200、201；messages 只覆盖 user 内容长度 201"""
    mock_create = mocker.patch("openai.resources.chat.completions.Completions.create")
    reply = "长度校验通过"
    mock_create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(content=reply))]
    )
    too_long = length > MAX_PROMPT_LENGTH
    length_fields = {"prompt_len": length} if messages is None else {"messages_len": length}
    record_payload(
        prompt,
        None if too_long else reply,
        exc=r"^prompt长度不能超过" if too_long else None,
        messages=messages,
        case_title=case_title,
        **length_fields,
    )
    if too_long:
        with pytest.raises(ValueError, match=r"^prompt长度不能超过"):
            llm_client.chat(prompt=prompt, messages=messages)
        mock_create.assert_not_called()
        return
    result = llm_client.chat(prompt=prompt, messages=messages)
    assert result == reply
    mock_create.assert_called_once()
