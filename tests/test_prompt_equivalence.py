import pytest
from utils.llm_client import chat,MAX_PROMPT_LENGTH


# parametrize: 等价类测试
# 一次性传入多组prompt，自动生成多条测试用例
@pytest.mark.real_api
@pytest.mark.parametrize("prompt, expect_valid", [
    # ===== 有效等价类 =====
    ("解释pytest是什么", True),                 # 有效：正常中文文本
    ("!@#$%^&*()<>\"'\n\r\temoji😄", True),    # 有效：特殊字符+emoji
    # ===== 无效等价类 =====
    ("", False),                                # 无效：空字符串
    (None, False),                              # 无效：prompt为None
    ("  ", False),                               # 无效：全空白空格字符串
    ("a"*(MAX_PROMPT_LENGTH+5), False)          # 无效：长度长于200
])
def test_prompt_equivalence(prompt, expect_valid, request):
    # ========== 所有记录信息存入stash，给makereport钩子读取 ==========
    request.node.stash["payload_data"] = {
        "prompt": prompt,
        "prompt_len": len(prompt) if prompt is not None else None,
        "response": None,
        "response_len": None,
        "expected_exception": None
    }
    exc_info = None

    if expect_valid:
        res = chat(prompt)
        request.node.stash["payload_data"].update({
            "response": res,
            "response_len": len(res)
        })
        assert len(res.strip()) > 0
    elif prompt is None:
        with pytest.raises(ValueError, match="prompt不能为None")as exc_info:
            chat(prompt)
    elif not len(prompt.strip()):
        with pytest.raises(ValueError, match="prompt不能为空")as exc_info:
            chat(prompt)
    else:
        with pytest.raises(ValueError, match=r"^prompt长度不能超过") as exc_info:
            chat(prompt)
    # 成功捕获预期异常，更新异常信息
    if exc_info is not None:
        request.node.stash["payload_data"]["expected_exception"] = str(exc_info.value) 